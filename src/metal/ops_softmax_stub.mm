// This file contains the Metal stub for SoftMax::compute.
// It is intended to be included directly by src/ops/softmax.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/softmax.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void SoftMax::compute<Device::METAL, float>(
        const StorageView& input,
        const StorageView* bias,
        StorageView& output) const {
      (void)input; (void)bias; (void)output;
      throw std::runtime_error("SoftMax for Device::METAL (float) not implemented yet - included stub");
    }
  }
}
