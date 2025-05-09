#include "ctranslate2/ops/awq/dequantize_awq.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h"
#include <Eigen/Core> // For Eigen::half, if needed by other types
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void DequantizeAwq::dequantize<Device::METAL, int, half_float::half>(
        const StorageView& input,
        const StorageView& scale,
        const StorageView& zeros,
        StorageView& output) const {
      throw std::runtime_error("DequantizeAwq Op (int -> half_float::half) is not implemented for Metal backend");
    }

    // Add other specializations if linker errors appear for them.

  }
}
