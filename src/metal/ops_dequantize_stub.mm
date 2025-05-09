#include "ctranslate2/ops/dequantize.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void Dequantize::dequantize<Device::METAL, int8_t, float>(
        const StorageView& input,
        const StorageView& scale,
        StorageView& output) const {
      throw std::runtime_error("Dequantize Op (int8_t, float) is not implemented for Metal backend");
    }

    template<>
    void Dequantize::dequantize_gemm_output<Device::METAL, float>(
        const StorageView& c,
        const StorageView& a_scale,
        const StorageView& b_scale,
        const bool transpose_a,
        const bool transpose_b,
        const StorageView* bias,
        StorageView& y) const {
      throw std::runtime_error("Dequantize Op (dequantize_gemm_output<float>) is not implemented for Metal backend");
    }

    // Add other types if linker errors appear for them.

  }
}
