#include "ctranslate2/ops/bias_add.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void BiasAdd::compute<Device::METAL, float>(const StorageView& value,
                                                const StorageView& bias,
                                                StorageView& output) const {
      throw std::runtime_error("BiasAdd Op is not implemented for Metal backend (float)");
    }

    // Add other types like Eigen::half, bfloat16_t if linker errors appear for them.

  }
}
