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

#include "autoware/tensorrt_plugins/custom_unique_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>  // TODO(knzo25): delete this
#include <string>
#include <tuple>
#include <vector>

namespace nvinfer1::plugin
{

CustomUniquePlugin::CustomUniquePlugin(
  const std::string & name, CustomUniqueParameters const & params)
: layer_name_{name}, params_{params}
{
  initFieldsToSerialize();
}

void CustomUniquePlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * CustomUniquePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
  try {
    if (type == PluginCapabilityType::kBUILD) {
      return static_cast<IPluginV3OneBuild *>(this);
    }
    if (type == PluginCapabilityType::kRUNTIME) {
      return static_cast<IPluginV3OneRuntime *>(this);
    }
    PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
    return static_cast<IPluginV3OneCore *>(this);
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

IPluginV3 * CustomUniquePlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new CustomUniquePlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * CustomUniquePlugin::getPluginName() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_NAME;
}

char const * CustomUniquePlugin::getPluginVersion() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_VERSION;
}

char const * CustomUniquePlugin::getPluginNamespace() const noexcept
{
  return kCUSTOM_UNIQUE_PLUGIN_NAMESPACE;
}

std::int32_t CustomUniquePlugin::getNbOutputs() const noexcept
{
  // return 4;
  return 5;
}

std::int32_t CustomUniquePlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 1);
  // PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(in[0].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[1].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[2].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[3].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(out[4].desc.dims.nbDims == 0);

  PLUGIN_ASSERT(out[0].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[1].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[2].desc.type == in[0].desc.type);
  PLUGIN_ASSERT(out[3].desc.type == in[0].desc.type);

  return 0;
}

bool CustomUniquePlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  // PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(num_outputs == 5);

  return (
    in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR &&
    in_out[pos].desc.type == nvinfer1::DataType::kINT64);
}

std::int32_t CustomUniquePlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  // PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(num_outputs == 5);

  output_types[0] = input_types[0];
  output_types[1] = input_types[0];
  output_types[2] = input_types[0];
  output_types[3] = input_types[0];
  output_types[4] = input_types[0];

  return 0;
}

std::int32_t CustomUniquePlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs, IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 1);
  // PLUGIN_ASSERT(num_outputs == 4);
  PLUGIN_ASSERT(num_outputs == 5);
  PLUGIN_ASSERT(inputs[0].nbDims == 1);

  outputs[0].nbDims = 1;
  outputs[0].d[0] = expr_builder.declareSizeTensor(4, *inputs[0].d[0], *inputs[0].d[0]);

  outputs[1] = inputs[0];
  outputs[2] = inputs[0];

  outputs[3].nbDims = 1;
  outputs[3].d[0] = expr_builder.declareSizeTensor(4, *inputs[0].d[0], *inputs[0].d[0]);

  // num_activate_out
  outputs[4].nbDims = 0;

  return 0;
}

std::int32_t CustomUniquePlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  std::cout << "CustomUniquePlugin::enqueue" << std::endl;
  (void)input_desc;
  (void)output_desc;
  (void)inputs;
  (void)outputs;
  (void)workspace;
  (void)stream;

  return 0;
}

std::int32_t CustomUniquePlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * CustomUniquePlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * CustomUniquePlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t CustomUniquePlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace nvinfer1::plugin
