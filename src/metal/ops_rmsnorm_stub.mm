// This file contains the Metal stub for RMSNorm::compute.
// It is intended to be included directly by src/ops/rms_norm.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/rms_norm.h" // For RMSNorm class definition
#include "ctranslate2/devices.h"      // For Device::METAL
#include "ctranslate2/storage_view.h"
#include <stdexcept>                // For std::runtime_error

namespace ctranslate2 {
  namespace ops {

    // Metal specialization for RMSNorm::compute
    template<>
    void RMSNorm::compute<Device::METAL, float>(
        const StorageView& gamma,
        const StorageView& input,
        StorageView& output) const {
      // TODO: Implement Metal RMSNorm
      (void)gamma; (void)input; (void)output;
      throw std::runtime_error("RMSNorm for Device::METAL (float) not implemented yet - included stub");
    }

  } // namespace ops
} // namespace ctranslate2
