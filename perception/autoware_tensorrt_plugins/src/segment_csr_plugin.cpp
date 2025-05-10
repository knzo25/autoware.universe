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

#include "autoware/tensorrt_plugins/segment_csr_plugin.hpp"

#include "autoware/scatter_ops/reduction.h"
#include "autoware/scatter_ops/segment_csr.h"
#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <algorithm>  // TODO(knzo25): delete this
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>  // TODO(knzo25): delete this
#include <memory>
#include <string>
#include <tuple>  // TODO(knzo25): delete this
#include <unordered_map>
#include <vector>
namespace nvinfer1::plugin
{

SegmentCSRPlugin::SegmentCSRPlugin(const std::string & name, const std::string & reduce)
: layer_name_{name}, reduce_{reduce}
{
  /* std::cout << "SegmentCSRPlugin::SegmentCSRPlugin | name: " << name << std::endl;
  std::cout << "SegmentCSRPlugin::SegmentCSRPlugin | reduce: " << reduce_ << std::endl; */
  initFieldsToSerialize();
}

void SegmentCSRPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back(
    "reduce", reduce_.c_str(), PluginFieldType::kCHAR, reduce_.size());

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * SegmentCSRPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 * SegmentCSRPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new SegmentCSRPlugin{layer_name_, reduce_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * SegmentCSRPlugin::getPluginName() const noexcept
{
  return kSEGMENT_CSR_PLUGIN_NAME;
}

char const * SegmentCSRPlugin::getPluginVersion() const noexcept
{
  return kSEGMENT_CSR_PLUGIN_VERSION;
}

char const * SegmentCSRPlugin::getPluginNamespace() const noexcept
{
  return kSEGMENT_CSR_PLUGIN_NAMESPACE;
}

std::int32_t SegmentCSRPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t SegmentCSRPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);

  PLUGIN_ASSERT(in[INOUT_IN_SRC_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_IN_INDPTR_INDEX].desc.dims.nbDims == 1);

  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);

  PLUGIN_ASSERT(in[INOUT_IN_SRC_INDEX].desc.type == out[0].desc.type);

  return 0;
}

bool SegmentCSRPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);

  bool supported = in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;

  switch (pos) {
    case INOUT_IN_SRC_INDEX:
      supported &=
        (in_out[pos].desc.type == nvinfer1::DataType::kFLOAT ||
         in_out[pos].desc.type == nvinfer1::DataType::kHALF);
      break;
    case INOUT_IN_INDPTR_INDEX:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kINT64;
      break;
    case INOUT_OUT_INDEX:
      supported &= in_out[pos].desc.type == in_out[INOUT_IN_SRC_INDEX].desc.type;
      break;
    default:
      supported = false;
      break;
  }

  return supported;
}

std::int32_t SegmentCSRPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);

  output_types[0] = input_types[0];

  return 0;
}

std::int32_t SegmentCSRPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 2);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);
  PLUGIN_ASSERT(inputs[1].nbDims == 1);

  outputs[0].nbDims = 2;
  // outputs[0].d[0] = inputs[1].d[0] - 1;
  outputs[0].d[0] =
    expr_builder.operation(DimensionOperation::kSUB, *inputs[1].d[0], *expr_builder.constant(1));
  outputs[0].d[1] = inputs[0].d[1];

  return 0;
}

void _write_vector_to_text_file(
  std::string const & file_name, const std::vector<std::int64_t> & data)
{
  std::ofstream file(file_name);
  if (file.is_open()) {
    for (const auto & value : data) {
      file << value << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << file_name << std::endl;
  }
}

void _write_vector_to_text_file(std::string const & file_name, const std::vector<float> & data)
{
  std::ofstream file(file_name);
  if (file.is_open()) {
    for (const auto & value : data) {
      file << value << "\n";
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << file_name << std::endl;
  }
}

std::int32_t SegmentCSRPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  /* std::cout << "SegmentCSRPlugin::enqueue::start" << std::endl;
  std::cout << "layer_name_: " << layer_name_ << std::endl;

  std::string layer_name2 = layer_name_;
  std::replace(layer_name2.begin(), layer_name2.end(), '/', '_');
  std::transform(layer_name2.begin(), layer_name2.end(), layer_name2.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  std::cout << "layer_name2: " << layer_name2 << std::endl;

  std::cout << "input_desc[0]: nbDims: " << input_desc[0].dims.nbDims << " ["
            << input_desc[0].dims.d[0] << ", " << input_desc[0].dims.d[1] << "]" << std::endl;
  std::cout << "input_desc[1]: nbDims: " << input_desc[1].dims.nbDims << " ["
            << input_desc[1].dims.d[0] << "]" << std::endl;

  std::cout << "output_desc[0]: nbDims: " << output_desc[0].dims.nbDims << " ["
            << output_desc[0].dims.d[0] << ", " << output_desc[0].dims.d[1] << "]" << std::endl; */

  // auto num_output_bytes = output_desc[0].dims.d[0] * output_desc[0].dims.d[1] * sizeof(float);

  std::vector<int32_t> src_size{
    static_cast<int32_t>(input_desc[0].dims.d[0]), static_cast<int32_t>(input_desc[0].dims.d[1])};
  std::vector<int32_t> indptr_size{static_cast<int32_t>(input_desc[1].dims.d[0])};

  int32_t result;

  if (input_desc[0].type == nvinfer1::DataType::kFLOAT) {
    const float * src_ptr = reinterpret_cast<const float *>(inputs[0]);
    const int64_t * indptr_ptr = reinterpret_cast<const int64_t *>(inputs[1]);

    std::tuple<float *, int64_t *> out = std::make_tuple(static_cast<float *>(outputs[0]), nullptr);

    AT_DISPATCH_REDUCTION_TYPES(reduce_, [&] {
      result =
        segment_csr_launch<float, REDUCE>(src_ptr, src_size, indptr_ptr, indptr_size, out, stream);
    });
  } else if (input_desc[0].type == nvinfer1::DataType::kHALF) {
    const half * src_ptr = reinterpret_cast<const half *>(inputs[0]);
    const int64_t * indptr_ptr = reinterpret_cast<const int64_t *>(inputs[1]);

    std::tuple<half *, int64_t *> out = std::make_tuple(static_cast<half *>(outputs[0]), nullptr);

    AT_DISPATCH_REDUCTION_TYPES(reduce_, [&] {
      result =
        segment_csr_launch<half, REDUCE>(src_ptr, src_size, indptr_ptr, indptr_size, out, stream);
    });
  }

  /* std::cout << "segment_csr_launch result: " << result << std::endl;

  // Copy the inputs to host to check the result.
  std::vector<float> input_data(input_desc[0].dims.d[0] * input_desc[0].dims.d[1]);
  cudaMemcpyAsync(
    input_data.data(), inputs[0], input_desc[0].dims.d[0] * input_desc[0].dims.d[1] * sizeof(float),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << "input_data: ";
  // Print the first 6 feature of the first 20 points.
  for (std::int32_t i = 0; i < 20; ++i) {
    std::cout << "[";
    for (std::int32_t j = 0; j < 6; ++j) {
      std::cout << input_data[i * input_desc[0].dims.d[1] + j] << ", ";
    }
    std::cout << "]" << std::endl;
  }
  // Copy the indptr to host to check the result.
  std::vector<int64_t> indptr_data(input_desc[1].dims.d[0]);
  cudaMemcpyAsync(
    indptr_data.data(), inputs[1], input_desc[1].dims.d[0] * sizeof(int64_t),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << "indptr_data: ";
  // Print the first 6 feature of the first 20 points.
  for (std::int32_t i = 0; i < 20; ++i) {
    std::cout << indptr_data[i] << ", ";
  }
  std::cout << std::endl;

  // Copy the output to host to check the result.
  std::vector<float> output_data(output_desc[0].dims.d[0] * output_desc[0].dims.d[1]);
  cudaMemcpyAsync(output_data.data(), outputs[0], num_output_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << "output_data: \n";
  // Print the first 6 feature of the first 20 points.
  for (std::int32_t i = 0; i < 20; ++i) {
    std::cout << "[";
    for (std::int32_t j = 0; j < 6; ++j) {
      std::cout << output_data[i * output_desc[0].dims.d[1] + j] << ", ";
    }
    std::cout << "]" << std::endl;
  } */

  /* _write_vector_to_text_file(layer_name2 + "_input_data.txt", input_data);
  _write_vector_to_text_file(layer_name2 + "_indptr_data.txt", indptr_data);
  _write_vector_to_text_file(layer_name2 + "_output_data.txt", output_data); */

  (void)input_desc;
  (void)output_desc;
  (void)inputs;
  (void)outputs;
  (void)workspace;
  (void)stream;
  /* std::cout << "Name: " << layer_name_ << std::endl;
  std::cout << "Reduction type: " << reduce_ << " size=" << reduce_.size() << std::endl;

  // Check the contents of the map
  for (const auto & pair : reduce2REDUCE) {
    std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
  }

  std::cout << "Enum type: " << reduce2REDUCE.at(reduce_) << std::endl; */
  /* std::cout << "SegmentCSRPlugin::enqueue::end" << std::endl; */
  return 0;
}

std::int32_t SegmentCSRPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * SegmentCSRPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * SegmentCSRPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t SegmentCSRPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace nvinfer1::plugin
