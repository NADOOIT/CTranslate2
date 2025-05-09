#include "ctranslate2/ops/mean.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h" // For bfloat16_t, etc.
#include <Eigen/Core>         // For Eigen::half
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void Mean::compute<Device::METAL, float>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t axis_size,
        const dim_t inner_size,
        const bool get_sum,
        StorageView& output) const {
      throw std::runtime_error("Mean Op (float) is not implemented for Metal backend");
    }

    // Add other specializations if linker errors appear for them.
    // Common types might be Eigen::half, bfloat16_t.

  }
}
