// This file contains the Metal stub for TopPMask::compute.
// It is intended to be included directly by src/ops/topp_mask.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/topp_mask.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void TopPMask::compute<Device::METAL, float>(
        const StorageView& probs,
        const StorageView& topp_params,
        StorageView& output) const {
      (void)probs; (void)topp_params; (void)output;
      throw std::runtime_error("TopPMask for Device::METAL (float) not implemented yet - included stub");
    }
  }
}

namespace ctranslate2 {
  namespace ops {
    template<>
    dim_t TopPMask::max_num_classes<Device::METAL>() {
      throw std::runtime_error("TopPMask::max_num_classes is not implemented for Metal backend");
    }
  }
}

