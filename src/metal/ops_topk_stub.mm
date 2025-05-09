#include "ctranslate2/ops/topk.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void TopK::compute<Device::METAL, float, int32_t>(
        const StorageView& x,
        StorageView& values,
        StorageView& indices) const {
      throw std::runtime_error("TopK Op (float, int32_t) is not implemented for Metal backend");
    }

    // Add other types if linker errors appear for them.

  }
}
