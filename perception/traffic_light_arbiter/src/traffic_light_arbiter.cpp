// Copyright 2023 The Autoware Contributors
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

#include "traffic_light_arbiter.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>

#include <lanelet2_core/primitives/BasicRegulatoryElements.h>

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include <rclcpp/time.hpp>

namespace lanelet
{

using TrafficLightConstPtr = std::shared_ptr<const TrafficLight>;

std::vector<TrafficLightConstPtr> filter_traffic_signals(const LaneletMapConstPtr map)
{
  std::vector<TrafficLightConstPtr> signals;
  for (const auto & element : map->regulatoryElementLayer) {
    const auto signal = std::dynamic_pointer_cast<const TrafficLight>(element);
    if (signal) {
      signals.push_back(signal);
    }
  }
  return signals;
}

}  // namespace lanelet

TrafficLightArbiter::TrafficLightArbiter(const rclcpp::NodeOptions & options)
: Node("traffic_light_selector", options)
{
  v2x_time_tolerance_ = this->declare_parameter<bool>("v2x_time_tolerance", false);
  perception_time_tolerance_ = this->declare_parameter<bool>("perception_time_tolerance", false);

  map_sub_ = create_subscription<LaneletMapBin>(
    "~/sub/vector_map", rclcpp::QoS(1).transient_local(),
    std::bind(&TrafficLightArbiter::onMap, this, std::placeholders::_1));

  perception_tlr_sub_ = create_subscription<TrafficLightArray>(
    "~/sub/perception_traffic_lights", rclcpp::QoS(1),
    std::bind(&TrafficLightArbiter::onPerceptionMsg, this, std::placeholders::_1));

  v2x_tlr_sub_ = create_subscription<TrafficLightArray>(
    "~/sub/v2x_traffic_lights", rclcpp::QoS(1),
    std::bind(&TrafficLightArbiter::onV2xMsg, this, std::placeholders::_1));

  pub_ = create_publisher<TrafficSignalArray>("~/pub/traffic_signals", rclcpp::QoS(1));
}

void TrafficLightArbiter::onMap(const LaneletMapBin::ConstSharedPtr msg)
{
  const auto map = std::make_shared<lanelet::LaneletMap>();
  lanelet::utils::conversion::fromBinMsg(*msg, map);

  const auto signals = lanelet::filter_traffic_signals(map);
  mapping_.clear();
  for (const auto & signal : signals) {
    for (const auto & light : signal->trafficLights()) {
      mapping_[light.id()] = signal->id();
      RCLCPP_WARN(get_logger(), "light=%d -> signal=%d", static_cast<int>(light.id()), static_cast<int>(signal->id()));
    }
  }
}

void TrafficLightArbiter::onPerceptionMsg(const TrafficLightArray::ConstSharedPtr msg)
{
  latest_perception_msg_ = *msg;

  if ((rclcpp::Time(msg->stamp) - rclcpp::Time(latest_v2x_msg_.stamp)).seconds() > v2x_time_tolerance_) {
    latest_v2x_msg_.lights.clear();    
  }

  arbiterAndPublish(msg->stamp);
}

void TrafficLightArbiter::onV2xMsg(const TrafficLightArray::ConstSharedPtr msg)
{
  latest_v2x_msg_ = *msg;

  if ((rclcpp::Time(msg->stamp) - rclcpp::Time(latest_perception_msg_.stamp)).seconds() > perception_time_tolerance_) {
    latest_v2x_msg_.lights.clear();    
  }

  arbiterAndPublish(msg->stamp);  
}

void TrafficLightArbiter::arbiterAndPublish(const builtin_interfaces::msg::Time & stamp)
{
  using TrafficSignal = autoware_perception_msgs::msg::TrafficSignal;
  using Element = autoware_perception_msgs::msg::TrafficLightElement;
  using ElementAndId = std::pair<Element, lanelet::Id>;
  
  std::unordered_map<lanelet::Id, std::vector<ElementAndId>> regulatory_element_lights_map;

    // Wait for vector map to create id mapping.
  if (mapping_.empty()) {
    RCLCPP_WARN(get_logger(), "Received light traffic messages before a map");
    return;
  }

  // Create function to add
  auto add_light_function = [&](const auto & light){
    const auto id = light.traffic_light_id;
    if (!mapping_.count(id)) {
      return;
    }
    auto & elements = regulatory_element_lights_map[mapping_[id]];
    for (const auto & element : light.elements) {
      elements.emplace_back(element, id);
    }
  };


  for (const auto & light : latest_perception_msg_.lights) {
    add_light_function(light);
  }

  for (const auto & light : latest_v2x_msg_.lights) {
    add_light_function(light);
  }

  // Use the most confident traffic light element in the same state.
  const auto get_highest_confidence_elements = [](const std::vector<ElementAndId> & element_and_id_vector) {
    using Key = std::tuple<Element::_color_type, Element::_shape_type>;
    std::map<Key, ElementAndId> highest_score_element_map;

    for (const auto & element_and_id : element_and_id_vector) {
      const auto key = std::make_tuple(element_and_id.first.color, element_and_id.first.shape);
      auto [iter, success] = highest_score_element_map.try_emplace(key, element_and_id);
      auto & iter_element_and_id = std::get<1>(*iter);
      if (!success && iter_element_and_id.first.confidence < element_and_id.first.confidence) {
        iter_element_and_id = element_and_id;
      }
    }

    std::unordered_map<lanelet::Id, TrafficSignal> result_signal_map;
    for (const auto & [k, v] : highest_score_element_map) {
      auto & signal = result_signal_map[v.second];
      signal.traffic_signal_id = v.second;
      signal.elements.push_back(v.first);
    }

    std::vector<TrafficSignal> result_signal_vector;
    result_signal_vector.resize(result_signal_map.size());

    for (const auto & [k, result_signal] : result_signal_map) {
      result_signal_vector.push_back(result_signal);
    }

    return result_signal_vector;
  };

  TrafficSignalArray array;
  array.stamp = stamp;
  for (const auto & [regulatory_element_id, elements] : regulatory_element_lights_map) {
    std::vector signal_msgs = get_highest_confidence_elements(elements);
    array.signals.insert(array.signals.end(), signal_msgs.begin(), signal_msgs.end());
  }

  pub_->publish(array);
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(TrafficLightArbiter)
