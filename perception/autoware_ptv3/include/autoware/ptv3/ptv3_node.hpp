// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__PTV3__PTV3_NODE_HPP_
#define AUTOWARE__PTV3__PTV3_NODE_HPP_

#include "autoware/ptv3/ptv3_trt.hpp"
#include "autoware/ptv3/visibility_control.hpp"

#include <Eigen/Core>
#include <autoware/universe_utils/ros/debug_publisher.hpp>
#include <autoware/universe_utils/ros/published_time_publisher.hpp>
#include <autoware/universe_utils/system/stop_watch.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::ptv3
{

class PTV3_PUBLIC PTv3Node : public rclcpp::Node
{
public:
  using Matrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  explicit PTv3Node(const rclcpp::NodeOptions & options);

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::ConstSharedPtr cloud_sub_{nullptr};
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_{nullptr};

  std::unique_ptr<PTv3TRT> model_ptr_{nullptr};

  // debugger
  std::unique_ptr<autoware::universe_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_{
    nullptr};
  std::unique_ptr<autoware::universe_utils::DebugPublisher> debug_publisher_ptr_{nullptr};
  std::unique_ptr<autoware::universe_utils::PublishedTimePublisher> published_time_pub_{nullptr};
};
}  // namespace autoware::ptv3

#endif  // AUTOWARE__PTV3__PTV3_NODE_HPP_
