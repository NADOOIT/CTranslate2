#include "ctranslate2/ops/rotary.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Rotary::compute<Device::METAL, float>(const StorageView& input,
                                               const StorageView& sin,
                                               const StorageView& cos,
                                               StorageView& output,
                                               bool is_transpose) const {
      throw std::runtime_error("Rotary Op is not implemented for Metal backend (float)");
    }

    // Add other types like Eigen::half, bfloat16_t if linker errors appear for them.

  }
}
