// Copyright 2021 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__LANE_CHANGE__UTIL_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__LANE_CHANGE__UTIL_HPP_

#include "behavior_path_planner/scene_module/lane_change/lane_change_module.hpp"
#include "behavior_path_planner/scene_module/lane_change/lane_change_path.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <route_handler/route_handler.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <lanelet2_core/primitives/Primitive.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace behavior_path_planner::lane_change_utils
{
using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::PredictedPath;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::Twist;
using tier4_autoware_utils::Polygon2d;

PathWithLaneId combineReferencePath(const PathWithLaneId path1, const PathWithLaneId path2);
bool isPathInLanelets(
  const PathWithLaneId & path, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets);
double getExpectedVelocityWhenDecelerate(
  const double & current_velocity, const double & expected_acceleration,
  const double & lane_change_prepare_duration);
double getDistanceWhenDecelerate(
  const double & velocity, const double & expected_acceleration, const double & duration,
  const double & minimum_distance);
std::vector<LaneChangePath> getLaneChangePaths(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & original_lanelets,
  const lanelet::ConstLanelets & target_lanelets, const Pose & pose, const Twist & twist,
  const BehaviorPathPlannerParameters & common_parameter, const LaneChangeParameters & parameter);
std::vector<LaneChangePath> selectValidPaths(
  const std::vector<LaneChangePath> & paths, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes,
  const lanelet::routing::RoutingGraphContainer & overall_graphs, const Pose & current_pose,
  const bool isInGoalRouteSection, const Pose & goal_pose);
bool selectSafePath(
  const std::vector<LaneChangePath> & paths, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes,
  const PredictedObjects::ConstSharedPtr dynamic_objects, const Pose & current_pose,
  const Twist & current_twist, const BehaviorPathPlannerParameters & common_parameters,
  const behavior_path_planner::LaneChangeParameters & ros_parameters,
  LaneChangePath * selected_path,
  std::unordered_map<std::string, CollisionCheckDebug> & debug_data);
bool isLaneChangePathSafe(
  const PathWithLaneId & path, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes,
  const PredictedObjects::ConstSharedPtr dynamic_objects, const Pose & current_pose,
  const Twist & current_twist, const BehaviorPathPlannerParameters & common_parameters,
  const behavior_path_planner::LaneChangeParameters & lane_change_parameters,
  std::unordered_map<std::string, CollisionCheckDebug> & debug_data, const bool use_buffer = true,
  const double acceleration = 0.0);
bool hasEnoughDistance(
  const LaneChangePath & path, const lanelet::ConstLanelets & current_lanes,
  const lanelet::ConstLanelets & target_lanes, const Pose & current_pose,
  const bool isInGoalRouteSection, const Pose & goal_pose,
  const lanelet::routing::RoutingGraphContainer & overall_graphs);
}  // namespace behavior_path_planner::lane_change_utils

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__LANE_CHANGE__UTIL_HPP_
