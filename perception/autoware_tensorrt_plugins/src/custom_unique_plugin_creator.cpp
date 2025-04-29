// Copyright 2025 TIER IV, Inc.
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

#include "autoware/tensorrt_plugins/custom_unique_plugin_creator.hpp"

#include "autoware/tensorrt_plugins//custom_unique_plugin.hpp"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntimePlugin.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace nvinfer1::plugin
{

REGISTER_TENSORRT_PLUGIN(CustomUniquePluginCreator);

CustomUniquePluginCreator::CustomUniquePluginCreator()
{
  plugin_attributes_.clear();
  fc_.nbFields = plugin_attributes_.size();
  fc_.fields = plugin_attributes_.data();
}

nvinfer1::PluginFieldCollection const * CustomUniquePluginCreator::getFieldNames() noexcept
{
  // This is only used in the build phase.
  return &fc_;
}

IPluginV3 * CustomUniquePluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc, TensorRTPhase phase) noexcept
{
  // The build phase and the deserialization phase are handled differently.
  if (phase == TensorRTPhase::kBUILD || phase == TensorRTPhase::kRUNTIME) {
    // The attributes from the ONNX node will be parsed and passed via fc.
    try {
      // nvinfer1::PluginField const * fields{fc->fields};
      std::int32_t num_fields{fc->nbFields};

      PLUGIN_VALIDATE(num_fields == 0);

      CustomUniqueParameters parameters;

      CustomUniquePlugin * const plugin{new CustomUniquePlugin{std::string(name), parameters}};
      return plugin;
    } catch (std::exception const & e) {
      caughtError(e);
    }
    return nullptr;
  } else if (phase == TensorRTPhase::kRUNTIME) {
    // The attributes from the serialized plugin will be passed via fc.
    try {
      nvinfer1::PluginField const * fields{fc->fields};
      std::int32_t num_fields{fc->nbFields};
      PLUGIN_VALIDATE(num_fields == 1);

      char const * attr_name = fields[0].name;
      PLUGIN_VALIDATE(!strcmp(attr_name, "parameters"));
      PLUGIN_VALIDATE(fields[0].type == nvinfer1::PluginFieldType::kUNKNOWN);
      PLUGIN_VALIDATE(fields[0].length == sizeof(CustomUniqueParameters));
      CustomUniqueParameters params{*(static_cast<CustomUniqueParameters const *>(fields[0].data))};

      CustomUniquePlugin * const plugin{new CustomUniquePlugin{std::string(name), params}};
      return plugin;
    } catch (std::exception const & e) {
      caughtError(e);
    }
    return nullptr;
  } else {
    return nullptr;
  }
}

}  // namespace nvinfer1::plugin
