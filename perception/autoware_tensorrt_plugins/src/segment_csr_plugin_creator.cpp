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

#include "autoware/tensorrt_plugins/segment_csr_plugin_creator.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"
#include "autoware/tensorrt_plugins/segment_csr_plugin.hpp"

#include <NvInferRuntimePlugin.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

namespace nvinfer1::plugin
{

REGISTER_TENSORRT_PLUGIN(SegmentCSRPluginCreator);

SegmentCSRPluginCreator::SegmentCSRPluginCreator()
{
  plugin_attributes_.clear();
  plugin_attributes_.emplace_back("reduce", nullptr, PluginFieldType::kCHAR, 1);

  fc_.nbFields = plugin_attributes_.size();
  fc_.fields = plugin_attributes_.data();
}

nvinfer1::PluginFieldCollection const * SegmentCSRPluginCreator::getFieldNames() noexcept
{
  // This is only used in the build phase.
  return &fc_;
}

IPluginV3 * SegmentCSRPluginCreator::createPlugin(
  char const * name, PluginFieldCollection const * fc,
  [[maybe_unused]] TensorRTPhase phase) noexcept
{
  try {
    PLUGIN_VALIDATE(fc != nullptr);
    PLUGIN_VALIDATE(fc->nbFields == 1);
    PLUGIN_VALIDATE(std::string(fc->fields[0].name) == "reduce");
    std::string reduce(static_cast<char const *>(fc->fields[0].data), fc->fields[0].length);

    std::size_t null_pos = reduce.find('\0');
    if (null_pos != std::string::npos) {
      reduce.resize(null_pos);
    }

    SegmentCSRPlugin * const plugin(new SegmentCSRPlugin(std::string(name), reduce));
    return plugin;
  } catch (std::exception & e) {
    caughtError(e);
  }
  return nullptr;

  /*

    // The build phase and the deserialization phase are handled differently.
    if (phase == TensorRTPhase::kBUILD) {
      // The attributes from the ONNX node will be parsed and passed via fc.
      try {
        nvinfer1::PluginField const * fields{fc->fields};
        std::int32_t num_fields{fc->nbFields};

        PLUGIN_VALIDATE(num_fields == 1);

        SegmentCSRParameters parameters;

        const std::string attr_name = fields[0].name;
        const nvinfer1::PluginFieldType type = fields[0].type;

        if (attr_name == "reduce") {
          PLUGIN_VALIDATE(type == nvinfer1::PluginFieldType::kCHAR);
          parameters.reduce = std::string(static_cast<char const *>(fields[0].data));
        }

        // Log the attributes parsed from ONNX node.
        std::stringstream ss;
        ss << name << " plugin Attributes:";
        logDebug(ss.str().c_str());

        ss.str("");
        ss << "reduce: " << parameters.reduce;
        logDebug(ss.str().c_str());

        SegmentCSRPlugin * const plugin{new SegmentCSRPlugin{std::string(name), parameters}};
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
        PLUGIN_VALIDATE(fields[0].length == sizeof(SegmentCSRParameters));
        SegmentCSRParameters params{*(static_cast<SegmentCSRParameters const *>(fields[0].data))};

        SegmentCSRPlugin * const plugin{new SegmentCSRPlugin{std::string(name), params}};
        return plugin;
      } catch (std::exception const & e) {
        caughtError(e);
      }
      return nullptr;
    } else {
      return nullptr;
    } */
}

}  // namespace nvinfer1::plugin
