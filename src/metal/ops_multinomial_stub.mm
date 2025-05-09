#include "ctranslate2/ops/multinomial.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Multinomial::compute<Device::METAL, float>(
        const StorageView& input,
        StorageView& output) const {
      throw std::runtime_error("Multinomial Op (float) is not implemented for Metal backend");
    }

    // Add other types if linker errors appear for them.

  }
}
