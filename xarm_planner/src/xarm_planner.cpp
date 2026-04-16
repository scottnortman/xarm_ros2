/* Copyright 2021 UFACTORY Inc. All Rights Reserved.
 *
 * Software License Agreement (BSD License)
 *
 * Author: Vinman <vinman.cub@gmail.com>
 ============================================================================*/
 
#include "xarm_planner/xarm_planner.h"
#include <chrono>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>

namespace
{
const double kDefaultJumpThreshold = 0.0;
const double kDefaultEefStep = 0.005;
const double kDefaultMaxVelocityScalingFactor = 0.3;
const double kDefaultMaxAccelerationScalingFactor = 0.1;

double duration_seconds(const builtin_interfaces::msg::Duration& duration)
{
    return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1e-9;
}
}

namespace xarm_planner
{

XArmPlanner::XArmPlanner(const rclcpp::Node::SharedPtr& node, const std::string& group_name)
    : node_(node)
{
    init(group_name);
}

XArmPlanner::XArmPlanner(const std::string& group_name)
{
    node_ = rclcpp::Node::make_shared("xarm_planner_move_group_node");
    init(group_name);
}

void XArmPlanner::init(const std::string& group_name) 
{
    is_trajectory_ = false;
    group_name_ = group_name;
    node_->get_parameter_or("jump_threshold", jump_threshold_, kDefaultJumpThreshold);
    node_->get_parameter_or("eef_step", eef_step_, kDefaultEefStep);
    node_->get_parameter_or("max_velocity_scaling_factor", max_velocity_scaling_factor_, kDefaultMaxVelocityScalingFactor);
    node_->get_parameter_or("max_acceleration_scaling_factor", max_acceleration_scaling_factor_, kDefaultMaxAccelerationScalingFactor);
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, group_name);
    const bool state_monitor_ready = move_group_->startStateMonitor(1.0);
    RCLCPP_INFO(node_->get_logger(), "Planning frame: %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(node_->get_logger(), "End effector link: %s", move_group_->getEndEffectorLink().c_str());
    RCLCPP_INFO(node_->get_logger(), "Available Planning Groups:");
    std::copy(move_group_->getJointModelGroupNames().begin(), move_group_->getJointModelGroupNames().end(), std::ostream_iterator<std::string>(std::cout, ", "));
    move_group_->setMaxVelocityScalingFactor(max_velocity_scaling_factor_);
    move_group_->setMaxAccelerationScalingFactor(max_acceleration_scaling_factor_);
    if (state_monitor_ready)
        RCLCPP_INFO(node_->get_logger(), "Current-state monitor primed before planning.");
    else
        RCLCPP_WARN(node_->get_logger(), "Current-state monitor did not prime within 1.0 s; Cartesian retiming may fall back.");
    RCLCPP_INFO(
        node_->get_logger(),
        "Planner timing params: eef_step=%.4f, jump_threshold=%.3f, max_velocity_scaling_factor=%.3f, max_acceleration_scaling_factor=%.3f",
        eef_step_,
        jump_threshold_,
        max_velocity_scaling_factor_,
        max_acceleration_scaling_factor_);
}

bool XArmPlanner::planJointTarget(const std::vector<double>& joint_target)
{
    bool success = move_group_->setJointValueTarget(joint_target);
    if (!success)
        RCLCPP_WARN(node_->get_logger(), "setJointValueTarget: out of bounds");
    success = (move_group_->plan(xarm_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!success)
        RCLCPP_ERROR(node_->get_logger(), "planJointTarget: plan failed");
    is_trajectory_ = false;
    return success;
}

bool XArmPlanner::planPoseTarget(const geometry_msgs::msg::Pose& pose_target)
{
    bool success = move_group_->setPoseTarget(pose_target);
    if (!success)
        RCLCPP_WARN(node_->get_logger(), "setPoseTarget: out of bounds");
    success = (move_group_->plan(xarm_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!success)
        RCLCPP_ERROR(node_->get_logger(), "planPoseTarget: plan failed");
    is_trajectory_ = false;
    return success;
}

bool XArmPlanner::planPoseTargets(const std::vector<geometry_msgs::msg::Pose>& pose_target_vector)
{
    bool success = move_group_->setPoseTargets(pose_target_vector);
    if (!success)
        RCLCPP_WARN(node_->get_logger(), "setPoseTargets: out of bounds");
    success = (move_group_->plan(xarm_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!success)
        RCLCPP_ERROR(node_->get_logger(), "planPoseTargets: plan failed");
    is_trajectory_ = false;
    return success;
}

bool XArmPlanner::planCartesianPath(const std::vector<geometry_msgs::msg::Pose>& pose_target_vector)
{
    moveit::core::RobotState reference_state(move_group_->getRobotModel());
    bool have_reference_state = false;
    bool using_fallback_state = false;

    // Use the latest available robot state for retiming instead of waiting for
    // a state newer than "now". In the Webots fake stack, /joint_states can
    // legitimately lag the planner request time by a few milliseconds after the
    // arm settles, which makes MoveIt reject an otherwise usable state.
    if (const auto current_state = move_group_->getCurrentState(0.0)) {
        reference_state = *current_state;
        have_reference_state = true;
        move_group_->setStartState(reference_state);
    } else {
        move_group_->setStartStateToCurrentState();
    }

    double fraction = move_group_->computeCartesianPath(pose_target_vector, eef_step_, jump_threshold_, trajectory_);
    if(fraction < 0.9) {
        RCLCPP_ERROR(node_->get_logger(), "planCartesianPath: plan failed, fraction=%lf", fraction);
        return false;
    }

    if (!have_reference_state &&
        !trajectory_.joint_trajectory.joint_names.empty() &&
        !trajectory_.joint_trajectory.points.empty() &&
        trajectory_.joint_trajectory.points.front().positions.size() == trajectory_.joint_trajectory.joint_names.size()) {
        reference_state.setToDefaultValues();
        reference_state.setVariablePositions(
            trajectory_.joint_trajectory.joint_names,
            trajectory_.joint_trajectory.points.front().positions
        );
        reference_state.update();
        have_reference_state = true;
        using_fallback_state = true;
        RCLCPP_WARN(
            node_->get_logger(),
            "planCartesianPath: current-state monitor did not provide a usable start state; using the first Cartesian trajectory point as both the start-state and retiming reference"
        );
    }

    bool retimed = false;
    if (have_reference_state) {
        robot_trajectory::RobotTrajectory robot_trajectory(move_group_->getRobotModel(), group_name_);
        robot_trajectory.setRobotTrajectoryMsg(reference_state, trajectory_);

        trajectory_processing::TimeOptimalTrajectoryGeneration totg;
        retimed = totg.computeTimeStamps(
            robot_trajectory,
            max_velocity_scaling_factor_,
            max_acceleration_scaling_factor_
        );
        if (retimed) {
            robot_trajectory.getRobotTrajectoryMsg(trajectory_);
        } else {
            RCLCPP_WARN(node_->get_logger(), "planCartesianPath: trajectory retiming failed; executing unretimed Cartesian trajectory");
        }
    } else {
        RCLCPP_WARN(
            node_->get_logger(),
            "planCartesianPath: unable to obtain any reference state for retiming; executing unretimed Cartesian trajectory"
        );
    }

    const auto point_count = trajectory_.joint_trajectory.points.size();
    const double planned_duration = point_count > 0
        ? duration_seconds(trajectory_.joint_trajectory.points.back().time_from_start)
        : 0.0;
    RCLCPP_INFO(
        node_->get_logger(),
        "planCartesianPath: fraction=%.3f, input_waypoints=%zu, trajectory_points=%zu, planned_duration=%.3f s, retimed=%d, fallback_reference=%d, max_velocity_scaling_factor=%.3f, max_acceleration_scaling_factor=%.3f",
        fraction,
        pose_target_vector.size(),
        point_count,
        planned_duration,
        retimed,
        using_fallback_state,
        max_velocity_scaling_factor_,
        max_acceleration_scaling_factor_
    );

    is_trajectory_ = true;
    return true;
}

bool XArmPlanner::executePath(bool wait)
{
    const auto start = std::chrono::steady_clock::now();
    moveit::core::MoveItErrorCode code;
    if (wait)
        code = is_trajectory_ ? move_group_->execute(trajectory_) : move_group_->execute(xarm_plan_);
    else
        code =  is_trajectory_ ? move_group_->asyncExecute(trajectory_) : move_group_->asyncExecute(xarm_plan_);
    bool success = (code == moveit::core::MoveItErrorCode::SUCCESS);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (wait) {
        RCLCPP_INFO(
            node_->get_logger(),
            "executePath: wait=%d, is_trajectory=%d, wall_time=%.3f s, success=%d",
            wait,
            is_trajectory_,
            elapsed,
            success
        );
    }
    if (!success)
        RCLCPP_ERROR(node_->get_logger(), "executePath: execute failed, wait=%d, MoveItErrorCode=%d", wait, code.val);
    return success;
}
}
