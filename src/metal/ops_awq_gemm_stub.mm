// This file contains the Metal stubs for GemmAwq operations.
// It is intended to be included directly by src/ops/awq/gemm.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/awq/gemm.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void GemmAwq::compute<Device::METAL, half_float::half, int>(
        const StorageView& a, const StorageView& b, const StorageView& b_scale, const StorageView& b_zero,
        StorageView& c) const {
      (void)a; (void)b; (void)b_scale; (void)b_zero; (void)c;
      throw std::runtime_error("GemmAwq::compute for Device::METAL (half, int) not implemented yet - included stub");
    }
  }
}
