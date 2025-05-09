// This file contains the Metal stub for GELU::compute.
// It is intended to be included directly by src/ops/gelu.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/gelu.h" // For GELU class definition
#include "ctranslate2/devices.h"      // For Device::METAL
#include "ctranslate2/storage_view.h"

#include <stdexcept> // For std::runtime_error

namespace ctranslate2 {
  namespace ops {

    // Metal specialization for GELU::compute
    template<>
    void GELU::compute<Device::METAL, float>(
        const StorageView& x,
        StorageView& y) const {
      // TODO: Implement Metal GELU
      (void)x; (void)y;
      throw std::runtime_error("GELU for Device::METAL (float) not implemented yet - included stub");
    }

  } // namespace ops
} // namespace ctranslate2
