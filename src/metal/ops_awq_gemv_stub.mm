// This file contains the Metal stubs for GemvAwq operations.
// It is intended to be included directly by src/ops/awq/gemv.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/awq/gemv.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void GemvAwq::compute_gemv<Device::METAL, half_float::half, int>(
        const StorageView& a, const StorageView& b, const StorageView& b_scale, const StorageView& b_zero,
        StorageView& c) const {
      (void)a; (void)b; (void)b_scale; (void)b_zero; (void)c;
      throw std::runtime_error("GemvAwq::compute_gemv for Device::METAL (half, int) not implemented yet - included stub");
    }

    template<>
    void GemvAwq::compute_gemv2<Device::METAL, half_float::half, int>(
        const StorageView& a, const StorageView& b, const StorageView& b_scale, const StorageView& b_zero,
        StorageView& c) const {
      (void)a; (void)b; (void)b_scale; (void)b_zero; (void)c;
      throw std::runtime_error("GemvAwq::compute_gemv2 for Device::METAL (half, int) not implemented yet - included stub");
    }
  }
}
