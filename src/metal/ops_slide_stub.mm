#include "ctranslate2/ops/slide.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h" // For bfloat16_t, etc.
#include <Eigen/Core>         // For Eigen::half
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void Slide::compute<Device::METAL, float>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (float) is not implemented for Metal backend");
    }

    template<>
    void Slide::compute<Device::METAL, int8_t>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (int8_t) is not implemented for Metal backend");
    }

    template<>
    void Slide::compute<Device::METAL, int16_t>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (int16_t) is not implemented for Metal backend");
    }

    template<>
    void Slide::compute<Device::METAL, int32_t>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (int32_t) is not implemented for Metal backend");
    }

    template<>
    void Slide::compute<Device::METAL, half_float::half>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (half_float::half) is not implemented for Metal backend");
    }

    template<>
    void Slide::compute<Device::METAL, bfloat16_t>(
        const StorageView& input,
        StorageView& output,
        const dim_t& index) const {
      throw std::runtime_error("Slide Op (bfloat16_t) is not implemented for Metal backend");
    }

    // Add other specializations if linker errors appear for them.
    // Common types might be Eigen::half, bfloat16_t.

  }
}
