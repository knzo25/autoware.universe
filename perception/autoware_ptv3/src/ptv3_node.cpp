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

#include "autoware/ptv3/ptv3_node.hpp"

#include "autoware/ptv3/utils.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>  // TODO(knzo25): delete this
#include <vector>

namespace autoware::ptv3
{

PTv3Node::PTv3Node(const rclcpp::NodeOptions & options) : Node("ptv3", options)
{
  // sleep 10 to attach the debugger
  // rclcpp::sleep_for(std::chrono::seconds(20));

  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{}.set__read_only(true);

  // TensorRT parameters
  const std::string plugins_path = this->declare_parameter<std::string>("plugins_path", descriptor);

  // Network parameters
  const std::string onnx_path = this->declare_parameter<std::string>("onnx_path", descriptor);
  const std::string engine_path = this->declare_parameter<std::string>("engine_path", descriptor);
  const std::string trt_precision =
    this->declare_parameter<std::string>("trt_precision", descriptor);

  auto to_float_vector = [](const auto & v) -> std::vector<float> {
    return std::vector<float>(v.begin(), v.end());
  };

  // Lidar branch parameters
  const auto cloud_capacity = this->declare_parameter<std::int64_t>("cloud_capacity", descriptor);

  const auto voxels_num =
    this->declare_parameter<std::vector<std::int64_t>>("voxels_num", descriptor);
  const auto point_cloud_range =
    to_float_vector(this->declare_parameter<std::vector<double>>("point_cloud_range", descriptor));
  const auto voxel_size =
    to_float_vector(this->declare_parameter<std::vector<double>>("voxel_size", descriptor));

  // Head parameters
  // class_names_ = this->declare_parameter<std::vector<std::string>>("class_names", descriptor);

  if (point_cloud_range.size() != 6) {
    RCLCPP_ERROR(rclcpp::get_logger("ptv3"), "The size of point_cloud_range != 6");

    throw std::runtime_error("The size of point_cloud_range != 6");
  }
  if (voxel_size.size() != 3) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("ptv3"), "The size of voxel_size != 3");

    throw std::runtime_error("The size of voxel_size != 3");
  }

  PTv3Config config(plugins_path, cloud_capacity, voxels_num, point_cloud_range, voxel_size);

  auto trt_config =
    tensorrt_common::TrtCommonConfig(onnx_path, trt_precision, engine_path, 1ULL << 33U);
  model_ptr_ = std::make_unique<PTv3TRT>(trt_config, config);

  cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "~/input/pointcloud", rclcpp::SensorDataQoS{}.keep_last(1),
    std::bind(&PTv3Node::cloudCallback, this, std::placeholders::_1));

  cloud_pub_ =
    this->create_publisher<sensor_msgs::msg::PointCloud2>("~/output/pointcloud", rclcpp::QoS(1));

  published_time_pub_ = std::make_unique<autoware::universe_utils::PublishedTimePublisher>(this);

  // initialize debug tool
  {
    using autoware::universe_utils::DebugPublisher;
    using autoware::universe_utils::StopWatch;
    stop_watch_ptr_ = std::make_unique<StopWatch<std::chrono::milliseconds>>();
    debug_publisher_ptr_ = std::make_unique<DebugPublisher>(this, this->get_name());
    stop_watch_ptr_->tic("cyclic");
    stop_watch_ptr_->tic("processing/total");
  }

  if (this->declare_parameter<bool>("build_only", false, descriptor)) {
    RCLCPP_INFO(this->get_logger(), "TensorRT engine was built. Shutting down the node.");
    rclcpp::shutdown();
  }
}

void PTv3Node::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pc_msg)
{
  const auto sub_count =
    cloud_pub_->get_subscription_count() + cloud_pub_->get_intra_process_subscription_count();
  if (sub_count < 1) {
    return;
  }

  if (stop_watch_ptr_) {
    stop_watch_ptr_->toc("processing/total", true);
  }

  auto output_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  std::unordered_map<std::string, double> proc_timing;
  bool is_success = model_ptr_->segment(pc_msg, *output_msg, proc_timing);
  if (!is_success) {
    return;
  }

  if (sub_count > 0) {
    cloud_pub_->publish(std::move(output_msg));
    published_time_pub_->publish_if_subscribed(cloud_pub_, output_msg->header.stamp);
  }

  // add processing time for debug
  if (debug_publisher_ptr_ && stop_watch_ptr_) {
    const double cyclic_time_ms = stop_watch_ptr_->toc("cyclic", true);
    const double processing_time_ms = stop_watch_ptr_->toc("processing/total", true);
    const double pipeline_latency_ms =
      std::chrono::duration<double, std::milli>(
        std::chrono::nanoseconds(
          (this->get_clock()->now() - output_msg->header.stamp).nanoseconds()))
        .count();
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/cyclic_time_ms", cyclic_time_ms);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/pipeline_latency_ms", pipeline_latency_ms);
    debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(
      "debug/processing_time/total_ms", processing_time_ms);
    for (const auto & [topic, time_ms] : proc_timing) {
      debug_publisher_ptr_->publish<tier4_debug_msgs::msg::Float64Stamped>(topic, time_ms);
    }
  }
}

}  // namespace autoware::ptv3

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(autoware::ptv3::PTv3Node)
