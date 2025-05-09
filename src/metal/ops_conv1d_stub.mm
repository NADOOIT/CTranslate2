#include "ctranslate2/ops/conv1d.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Conv1D::compute<Device::METAL, float>(const StorageView& input,
                                               const StorageView& weight,
                                               const StorageView* bias,
                                               StorageView& output,
                                               const StorageView* qscale) const {
      throw std::runtime_error("Conv1D Op is not implemented for Metal backend (float)");
    }

    // Add other types like Eigen::half, bfloat16_t if linker errors appear for them.

  }
}
