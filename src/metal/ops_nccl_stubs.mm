// This file contains the Metal stubs for GatherAll::compute and ReduceAll::compute.
// It is intended to be included directly by src/ops/nccl_ops.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/nccl_ops.h" // For GatherAll, ReduceAll class definitions
#include "ctranslate2/devices.h"      // For Device::METAL
#include "ctranslate2/storage_view.h"
#include "ctranslate2/types.h"        // For bfloat16_t
#include "half_float/half.hpp"      // For half_float::half
#include <stdexcept>                // For std::runtime_error

namespace ctranslate2 {
  namespace ops {

    // GatherAll Stubs for Metal
#define DECLARE_GATHER_ALL_STUB(T) \
    template<> \
    void GatherAll::compute<Device::METAL, T>(const StorageView& input, StorageView& output) const { \
      (void)input; (void)output; \
      throw std::runtime_error("GatherAll for Device::METAL (" #T ") not implemented yet - included stub"); \
    }
    DECLARE_GATHER_ALL_STUB(signed char)
    DECLARE_GATHER_ALL_STUB(float)
    DECLARE_GATHER_ALL_STUB(int)
    DECLARE_GATHER_ALL_STUB(short)
    DECLARE_GATHER_ALL_STUB(bfloat16_t)
    DECLARE_GATHER_ALL_STUB(half_float::half)
#undef DECLARE_GATHER_ALL_STUB

    // ReduceAll Stubs for Metal
#define DECLARE_REDUCE_ALL_STUB(T) \
    template<> \
    void ReduceAll::compute<Device::METAL, T>(const StorageView& input, StorageView& output) const { \
      (void)input; (void)output; \
      throw std::runtime_error("ReduceAll for Device::METAL (" #T ") not implemented yet - included stub"); \
    }
    DECLARE_REDUCE_ALL_STUB(half_float::half)
    DECLARE_REDUCE_ALL_STUB(bfloat16_t)
    DECLARE_REDUCE_ALL_STUB(signed char)
    DECLARE_REDUCE_ALL_STUB(float)
    DECLARE_REDUCE_ALL_STUB(int)
    DECLARE_REDUCE_ALL_STUB(short)
#undef DECLARE_REDUCE_ALL_STUB

  } // namespace ops
} // namespace ctranslate2
