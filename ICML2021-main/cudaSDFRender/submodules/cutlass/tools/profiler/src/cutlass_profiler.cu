/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief Execution environment
*/

#include <iostream>
#include <stdexcept>

// CUTLASS Library includes
#include "cutlass/library/library.h"
#include "cutlass/library/manifest.h"

// Profiler includes
#include "cutlass_profiler.h"
#include "gemm_operation_profiler.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

CutlassProfiler::CutlassProfiler(
  Options const &options
): 
  options_(options) {

  operation_profilers_.emplace_back(new GemmOperationProfiler); 
}

CutlassProfiler::~CutlassProfiler() {

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Execute the program
int CutlassProfiler::operator()() {

  if (options_.about.help) {
    if (options_.operation_kind == library::OperationKind::kInvalid) {
      print_usage_(std::cout);
    }
    else {
      for (auto & profiler : operation_profilers_) {
        if (profiler->kind() == options_.operation_kind) {
          profiler->print_usage(std::cout);
          profiler->print_examples(std::cout);
          return 0;
        }
      }
    }
    return 0;
  }
  else if (options_.about.version) {
    options_.about.print_version(std::cout);

    std::cout << std::endl;
    return 0;
  }
  else if (options_.about.device_info) {
    options_.device.print_device_info(std::cout);
    return 0;
  }

  if (options_.execution_mode == ExecutionMode::kProfile ||
    options_.execution_mode == ExecutionMode::kDryRun ||
    options_.execution_mode == ExecutionMode::kTrace) {

    // Profiles all operations
    profile_();
  }
  else if (options_.execution_mode == ExecutionMode::kEnumerate) {
    // Enumerates all operations
    enumerate_();
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerates all operations
void CutlassProfiler::enumerate_() {

}

/// Profiles all operations
int CutlassProfiler::profile_() {

  library::Manifest manifest;
  cutStatus status = manifest.initialize();

  if (status != cutStatus::kSuccess) {
    return -1;
  }

  int result = 0;
  DeviceContext device_context;

  // For all profilers
  for (auto & profiler : operation_profilers_) {

    if (options_.operation_kind == library::OperationKind::kInvalid ||
      options_.operation_kind == profiler->kind()) {

      result = profiler->profile_all(options_, manifest, device_context);

      if (result) {
        return result;
      } 
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints all options
void CutlassProfiler::print_usage_(std::ostream &out) {
  options_.print_usage(out);

  out << "\nOperations:\n\n";

  // For all profilers
  for (auto & profiler : operation_profilers_) {


    std::string kind_str = library::to_string(profiler->kind());

    size_t kAlignment = 40;
    size_t columns = 0;

    if (kind_str.size() < kAlignment) {
      columns = kAlignment - kind_str.size();
    }

    out << "     " << kind_str << std::string(columns, ' ') << profiler->description() << "\n";

  }

  out << "\n\nFor details about a particular function, specify the function name with --help.\n\nExample:\n\n"
    << "  $ cutlass_profiler --operation=Gemm --help\n\n";
}

/// Prints usage
void CutlassProfiler::print_options_(std::ostream &out) {
  options_.print_options(out);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Initializes the CUDA device
void CutlassProfiler::initialize_device_() {

  cudaError_t result = cudaSetDevice(options_.device.device);

  if (result != cudaSuccess) {
    std::cerr << "Failed to set device.";
    throw std::runtime_error("Failed to set device");
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
