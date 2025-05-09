// This file contains the Metal stub for GumbelMax::add_gumbel_noise.
// It is intended to be included directly by src/ops/gumbel_max.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/gumbel_max.h" // For GumbelMax class definition
#include "ctranslate2/devices.h"      // For Device::METAL
#include "ctranslate2/storage_view.h"
#include <stdexcept>                // For std::runtime_error

namespace ctranslate2 {
  namespace ops {

    // Metal specialization for GumbelMax::add_gumbel_noise
    template<>
    void GumbelMax::add_gumbel_noise<Device::METAL, float>(
        const StorageView& x,
        StorageView& y) const {
      // TODO: Implement Metal GumbelMax::add_gumbel_noise
      (void)x; (void)y;
      throw std::runtime_error("GumbelMax::add_gumbel_noise for Device::METAL (float) not implemented yet - included stub");
    }

  } // namespace ops
} // namespace ctranslate2
