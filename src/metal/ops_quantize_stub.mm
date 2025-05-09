// This file contains the Metal stub for Quantize::quantize.
// It is intended to be included directly by src/ops/quantize.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/quantize.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void Quantize::quantize<Device::METAL, float, signed char>(
        const StorageView& input,
        StorageView& output,
        StorageView& scale) const {
      (void)input; (void)output; (void)scale;
      throw std::runtime_error("Quantize for Device::METAL (float, signed char) not implemented yet - included stub");
    }
  }
}
