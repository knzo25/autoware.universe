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

#include "autoware/tensorrt_plugins/indice_conv_plugin.hpp"

#include "autoware/tensorrt_plugins/plugin_utils.hpp"

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>
#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>  // cSpell:ignore spconvlib
#include <spconvlib/spconv/csrc/sparse/alloc/StaticAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/spops/ConvGemmOps.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>

#include <algorithm>  // TODO(knzo25): delete this
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>  // TODO(knzo25): delete this
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

void loadIntegers(const std::string & filename, std::vector<int32_t> & integers)
{
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;

    char * endptr = nullptr;
    int32_t value = std::strtoll(line.c_str(), &endptr, 10);

    if (endptr != nullptr && *endptr == '\0') {
      integers.push_back(value);
    } else {
      // Not a valid integer line, skip
    }
  }
}

void loadFloats(const std::string & filename, std::vector<float> & floats)
{
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) continue;

    char * endptr = nullptr;
    float value = std::strtof(line.c_str(), &endptr);

    if (endptr != nullptr && *endptr == '\0') {
      floats.push_back(value);
    } else {
      // Not a valid float line, skip
    }
  }
}

namespace nvinfer1::plugin
{

IndiceConvPlugin::IndiceConvPlugin(const std::string & name, IndiceConvParameters const & params)
: layer_name_{name}, params_{params}
{
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;

  initFieldsToSerialize();

  arch_ = ConvGemmOps::get_compute_capability();
  tuner_fp16_ptr_ =
    std::make_unique<GemmTunerSimple>(GemmMain::get_all_algo_desp());  // cSpell:ignore desp
  tuner_fp32_ptr_ = std::make_unique<GemmTunerSimple>(GemmMain::get_all_algo_desp());
}

void IndiceConvPlugin::initFieldsToSerialize()
{
  data_to_serialize_.clear();
  data_to_serialize_.emplace_back(
    "is_subm", &params_.is_subm, PluginFieldType::kINT32, 1);  // cSpell:ignore subm

  fc_to_serialize_.nbFields = data_to_serialize_.size();
  fc_to_serialize_.fields = data_to_serialize_.data();
}

IPluginCapability * IndiceConvPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
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

IPluginV3 * IndiceConvPlugin::clone() noexcept
{
  try {
    IPluginV3 * const plugin{new IndiceConvPlugin{layer_name_, params_}};
    return plugin;
  } catch (std::exception const & e) {
    caughtError(e);
  }
  return nullptr;
}

char const * IndiceConvPlugin::getPluginName() const noexcept
{
  return kINDICE_CONV_PLUGIN_NAME;
}

char const * IndiceConvPlugin::getPluginVersion() const noexcept
{
  return kINDICE_CONV_PLUGIN_VERSION;
}

char const * IndiceConvPlugin::getPluginNamespace() const noexcept
{
  return kINDICE_CONV_PLUGIN_NAMESPACE;
}

std::int32_t IndiceConvPlugin::getNbOutputs() const noexcept
{
  return 1;
}

std::int32_t IndiceConvPlugin::configurePlugin(
  DynamicPluginTensorDesc const * in, std::int32_t num_inputs, DynamicPluginTensorDesc const * out,
  std::int32_t num_outputs) noexcept
{
  // Validate input arguments.
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(in[INOUT_FILTERS_INDEX].desc.dims.nbDims == 5);
  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_INDEX].desc.dims.nbDims == 3);
  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_NUM_INDEX].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(in[INOUT_NUM_ACTIVATE_OUT_INDEX].desc.dims.nbDims == 0);
  PLUGIN_ASSERT(out[0].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(
    in[INOUT_FILTERS_INDEX].desc.dims.d[4] == in[INOUT_IN_FEATURES_INDEX].desc.dims.d[1]);

  PLUGIN_ASSERT(in[INOUT_INDICE_PAIRS_INDEX].desc.dims.d[0] == 2);

  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == in[INOUT_FILTERS_INDEX].desc.type);
  PLUGIN_ASSERT(in[INOUT_IN_FEATURES_INDEX].desc.type == out[0].desc.type);
  PLUGIN_ASSERT(
    in[INOUT_INDICE_PAIRS_INDEX].desc.type == in[INOUT_INDICE_PAIRS_NUM_INDEX].desc.type);
  return 0;
}

bool IndiceConvPlugin::supportsFormatCombination(
  std::int32_t pos, DynamicPluginTensorDesc const * in_out, std::int32_t num_inputs,
  std::int32_t num_outputs) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);

  bool supported = in_out[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;

  switch (pos) {
    case INOUT_IN_FEATURES_INDEX:
      supported &=
        (in_out[pos].desc.type == nvinfer1::DataType::kFLOAT ||
         in_out[pos].desc.type == nvinfer1::DataType::kHALF);
      break;
    case INOUT_FILTERS_INDEX:
    case INOUT_OUT_FEATURES_INDEX:
      supported &= in_out[pos].desc.type == in_out[INOUT_IN_FEATURES_INDEX].desc.type;
      break;
    case INOUT_INDICE_PAIRS_INDEX:
    case INOUT_INDICE_PAIRS_NUM_INDEX:
    case INOUT_NUM_ACTIVATE_OUT_INDEX:
      supported &= in_out[pos].desc.type == nvinfer1::DataType::kINT32;
      break;
    default:
      supported = false;
      break;
  }

  return supported;
}

std::int32_t IndiceConvPlugin::getOutputDataTypes(
  DataType * output_types, std::int32_t num_outputs, DataType const * input_types,
  std::int32_t num_inputs) const noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);

  output_types[0] = input_types[INOUT_IN_FEATURES_INDEX];

  return 0;
}

std::int32_t IndiceConvPlugin::getOutputShapes(
  DimsExprs const * inputs, std::int32_t num_inputs,
  [[maybe_unused]] DimsExprs const * shape_inputs, [[maybe_unused]] std::int32_t num_shape_inputs,
  DimsExprs * outputs, std::int32_t num_outputs,
  [[maybe_unused]] IExprBuilder & expr_builder) noexcept
{
  PLUGIN_ASSERT(num_inputs == 5);
  PLUGIN_ASSERT(num_outputs == 1);
  PLUGIN_ASSERT(inputs[0].nbDims == 2);

  outputs[0].nbDims = 2;
  outputs[0].d[0] = inputs[INOUT_INDICE_PAIRS_INDEX].d[2];
  outputs[0].d[1] = inputs[INOUT_FILTERS_INDEX].d[0];

  return 0;
}

std::int32_t IndiceConvPlugin::enqueue(
  PluginTensorDesc const * input_desc, [[maybe_unused]] PluginTensorDesc const * output_desc,
  void const * const * inputs, void * const * outputs, [[maybe_unused]] void * workspace,
  cudaStream_t stream) noexcept
{
  /* std::cout << "IndiceConvPlugin::enqueue::start" << std::endl;

  std::vector<float> input_data(input_desc[0].dims.d[0] * input_desc[0].dims.d[1]);
  int32_t sample_features = std::min<int32_t>(input_desc[0].dims.d[1], 6);
  cudaMemcpyAsync(
    input_data.data(), inputs[0], input_desc[0].dims.d[0] * input_desc[0].dims.d[1] * sizeof(float),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  std::cout << "input_data: ";
  // Print the first 6 feature of the first 20 points.
  for (std::int32_t i = 0; i < 20; ++i) {
    std::cout << "[";
    for (std::int32_t j = 0; j < sample_features; ++j) {
      std::cout << input_data[i * input_desc[0].dims.d[1] + j] << ", ";
    }
    std::cout << "]" << std::endl;
  } */

  using StaticAllocator = spconvlib::spconv::csrc::sparse::alloc::StaticAllocator;
  using ConvGemmOps = spconvlib::spconv::csrc::sparse::convops::spops::ConvGemmOps;
  using SimpleExternalSpconvMatmul =
    spconvlib::spconv::csrc::sparse::convops::SimpleExternalSpconvMatmul;

  std::int64_t num_act_in = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[0];
  std::int64_t num_in_features = input_desc[INOUT_IN_FEATURES_INDEX].dims.d[1];
  // std::int64_t kernel_volume = input_desc[INOUT_PAIR_FWD_INDEX].dims.d[0];

  // TODO(knzo25): check if this is valid for the !subm case. We could use
  // INOUT_NUM_ACTIVATE_OUT_INDEX but that would require a sync
  std::int64_t num_act_out = input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2];
  std::int64_t num_out_features = input_desc[INOUT_FILTERS_INDEX].dims.d[0];

  auto in_features_type = input_desc[INOUT_IN_FEATURES_INDEX].type;
  [[maybe_unused]] auto filters_type = input_desc[INOUT_FILTERS_INDEX].type;
  [[maybe_unused]] auto out_features_type = input_desc[INOUT_OUT_FEATURES_INDEX].type;

  assert(in_features_type == filters_type);
  assert(in_features_type == out_features_type);

  auto dtype = in_features_type == DataType::kFLOAT ? tv::float32 : tv::float16;

  /** start of dummy input */
  /* std::vector<float> external_features_host;
  std::vector<float> external_weights_host;

  std::vector<std::int32_t> external_pairs_host;
  std::vector<std::int32_t> external_pairs_num_host; */

  /* loadFloats("indice_conv_features.txt", external_features_host);
  loadFloats("indice_conv_filters.txt", external_weights_host);
  loadIntegers("indice_conv_indice_pairs.txt", external_pairs_host);
  loadIntegers("indice_conv_indice_pair_num.txt", external_pairs_num_host);

  float * external_features_device;
  float * external_weights_device;
  int32_t * external_pairs_device;
  int32_t * external_pairs_num_device;
  cudaMalloc(
    reinterpret_cast<void **>(&external_features_device),
    external_features_host.size() * sizeof(float));
  cudaMalloc(
    reinterpret_cast<void **>(&external_weights_device),
    external_weights_host.size() * sizeof(float));
  cudaMalloc(
    reinterpret_cast<void **>(&external_pairs_device),
    external_pairs_host.size() * sizeof(int32_t));
  cudaMalloc(
    reinterpret_cast<void **>(&external_pairs_num_device),
    external_pairs_num_host.size() * sizeof(int32_t));

  cudaMemcpyAsync(
    external_features_device, external_features_host.data(),
    external_features_host.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    external_weights_device, external_weights_host.data(),
    external_weights_host.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    external_pairs_device, external_pairs_host.data(),
    external_pairs_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
    external_pairs_num_device, external_pairs_num_host.data(),
    external_pairs_num_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream); */

  /* tv::Tensor input_features =
    tv::from_blob(external_features_device, {num_act_in, num_in_features}, dtype, 0);

  tv::Tensor weights = tv::from_blob(
    external_weights_device,
    {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
     input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
     input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
    dtype, 0);

  tv::Tensor pairs = tv::from_blob(
    external_pairs_device,
    {input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[0], input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[1],
     input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2]},
    tv::int32, 0);

  tv::Tensor pairs_num = tv::from_blob(
    external_pairs_num_device, {input_desc[INOUT_INDICE_PAIRS_NUM_INDEX].dims.d[0]},
    tv::int32, 0); */

  tv::Tensor input_features =
    tv::from_blob(inputs[INOUT_IN_FEATURES_INDEX], {num_act_in, num_in_features}, dtype, 0);

  tv::Tensor input_features_fp32 =
    tv::from_blob(inputs[INOUT_IN_FEATURES_INDEX], {num_act_in, num_in_features}, tv::float32, 0);

  tv::Tensor weights = tv::from_blob(
    inputs[INOUT_FILTERS_INDEX],
    {input_desc[INOUT_FILTERS_INDEX].dims.d[0], input_desc[INOUT_FILTERS_INDEX].dims.d[1],
     input_desc[INOUT_FILTERS_INDEX].dims.d[2], input_desc[INOUT_FILTERS_INDEX].dims.d[3],
     input_desc[INOUT_FILTERS_INDEX].dims.d[4]},
    dtype, 0);

  tv::Tensor pairs = tv::from_blob(
    inputs[INOUT_INDICE_PAIRS_INDEX],
    {input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[0], input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[1],
     input_desc[INOUT_INDICE_PAIRS_INDEX].dims.d[2]},
    tv::int32, 0);

  tv::Tensor pairs_num = tv::from_blob(
    inputs[INOUT_INDICE_PAIRS_NUM_INDEX], {input_desc[INOUT_INDICE_PAIRS_NUM_INDEX].dims.d[0]},
    tv::int32, 0);

  tv::Tensor out_features = tv::from_blob(outputs[0], {num_act_out, num_out_features}, dtype, 0);

  auto & tuner_ptr = dtype == tv::float32 ? tuner_fp32_ptr_ : tuner_fp16_ptr_;

  // TODO(knzo25): Load txt data to check the correctness of the plugin

  // start

  if (params_.is_subm) {
    std::unordered_map<std::string, tv::Tensor> tensor_dict{
      {SPCONV_ALLOC_FEATURES, input_features},
      {SPCONV_ALLOC_FILTERS, weights},
      {SPCONV_ALLOC_OUT_FEATURES, out_features}};
    StaticAllocator alloc2(tensor_dict);

    SimpleExternalSpconvMatmul ext_mm(alloc2);

    ConvGemmOps::indice_conv(
      alloc2, ext_mm, *tuner_ptr, true, false, input_features, weights, pairs, pairs_num, arch_,
      out_features.dim(0), false, params_.is_subm,
      static_cast<int>(tv::gemm::SparseConvAlgo::kNative), reinterpret_cast<std::uintptr_t>(stream),
      tv::Tensor(), 0.f, 0.f, tv::gemm::Activation::kNone, false);
  } else {
    std::unordered_map<std::string, tv::Tensor> tensor_dict{
      {SPCONV_ALLOC_FEATURES, input_features},
      {SPCONV_ALLOC_FILTERS, weights},
      {SPCONV_ALLOC_OUT_FEATURES, out_features}};

    StaticAllocator alloc2(tensor_dict);

    SimpleExternalSpconvMatmul ext_mm(alloc2);
    ConvGemmOps::indice_conv(
      alloc2, ext_mm, *tuner_ptr, true, false, input_features, weights, pairs, pairs_num, arch_,
      out_features.dim(0), false, params_.is_subm,
      static_cast<int>(tv::gemm::SparseConvAlgo::kNative), reinterpret_cast<std::uintptr_t>(stream),
      tv::Tensor(), 0.f, 0.f, tv::gemm::Activation::kNone, false);
  }

  /* std::vector<float> weight_test_data(
    input_desc[INOUT_FILTERS_INDEX].dims.d[0] * input_desc[INOUT_FILTERS_INDEX].dims.d[1] *
    input_desc[INOUT_FILTERS_INDEX].dims.d[2] * input_desc[INOUT_FILTERS_INDEX].dims.d[3] *
    input_desc[INOUT_FILTERS_INDEX].dims.d[4]);
  cudaMemcpyAsync(
    weight_test_data.data(), inputs[INOUT_FILTERS_INDEX],
    input_desc[INOUT_FILTERS_INDEX].dims.d[0] * input_desc[INOUT_FILTERS_INDEX].dims.d[1] *
      input_desc[INOUT_FILTERS_INDEX].dims.d[2] * input_desc[INOUT_FILTERS_INDEX].dims.d[3] *
      input_desc[INOUT_FILTERS_INDEX].dims.d[4] * sizeof(float),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> out_features_host(num_act_out * num_out_features);
  cudaMemcpyAsync(
    out_features_host.data(), outputs[0], num_act_out * num_out_features * sizeof(float),
    cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::cout << "out_features_host: \n";
  int32_t sample_out_features = std::min<int32_t>(output_desc[0].dims.d[1], 6);
  // print all the out_features
  for (std::int32_t i = 0; i < 20; ++i) {
    std::cout << "[";
    for (std::int32_t j = 0; j < sample_out_features; ++j) {
      std::cout << out_features_host[i * num_out_features + j] << ", ";
    }
    std::cout << "]" << std::endl;
  } */

  /* std::cout << "weight_test_data: ";
  // print all the weights
  for(const auto & w : weight_test_data) {
    std::cout << w << ", ";
  }
  std::cout << std::endl; */

  /* std::cout << "input_features: " << input_features.cpu().slice_first_axis(0,20) << std::endl;
  std::cout << "weights: " <<
  weights.cpu().slice(1,0,1,1,false,false).slice(2,0,1,1,false,false).slice(3,0,1,1,false,false).slice(4,0,1,1,false,false)
  << std::endl; //
  //std::cout << "weights: " << weights.cpu() << std::endl;
  std::cout << "out_features: " <<
  out_features.cpu().slice_first_axis(0,20).slice(1,0,6,1,false,false) << std::endl; */

  /* std::cout << "IndiceConvPlugin::enqueue end" << std::endl; */

  return 0;
}

std::int32_t IndiceConvPlugin::onShapeChange(
  [[maybe_unused]] PluginTensorDesc const * in, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] PluginTensorDesc const * out, [[maybe_unused]] std::int32_t num_outputs) noexcept
{
  return 0;
}

IPluginV3 * IndiceConvPlugin::attachToContext(
  [[maybe_unused]] IPluginResourceContext * context) noexcept
{
  return clone();
}

PluginFieldCollection const * IndiceConvPlugin::getFieldsToSerialize() noexcept
{
  return &fc_to_serialize_;
}

std::size_t IndiceConvPlugin::getWorkspaceSize(
  [[maybe_unused]] DynamicPluginTensorDesc const * inputs, [[maybe_unused]] std::int32_t num_inputs,
  [[maybe_unused]] DynamicPluginTensorDesc const * outputs,
  [[maybe_unused]] std::int32_t num_outputs) const noexcept
{
  return 0;
}

}  // namespace nvinfer1::plugin
