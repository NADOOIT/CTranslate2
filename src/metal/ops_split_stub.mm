#include "ctranslate2/ops/split.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h"
#include <Eigen/Core> // For Eigen::half
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void Split::compute<Device::METAL, float>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (float) is not implemented for Metal backend");
    }

#ifdef CT2_WITH_HALF
    template<>
    void Split::compute<Device::METAL, Eigen::half>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (Eigen::half) is not implemented for Metal backend");
    }
#endif

    template<>
    void Split::compute<Device::METAL, half_float::half>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (half_float::half) is not implemented for Metal backend");
    }

#ifdef CT2_WITH_BFLOAT16
    template<>
    void Split::compute<Device::METAL, bfloat16_t>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (bfloat16_t) is not implemented for Metal backend");
    }
#endif

    template<>
    void Split::compute<Device::METAL, bfloat16_t>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      // This is the unconditional bfloat16_t from the linker error
      throw std::runtime_error("Split Op (unconditional bfloat16_t) is not implemented for Metal backend");
    }

    template<>
    void Split::compute<Device::METAL, int8_t>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (int8_t) is not implemented for Metal backend");
    }

    template<>
    void Split::compute<Device::METAL, int16_t>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (int16_t) is not implemented for Metal backend");
    }

    template<>
    void Split::compute<Device::METAL, int32_t>(
        const StorageView& input,
        std::vector<StorageView*>& outputs) const {
      throw std::runtime_error("Split Op (int32_t) is not implemented for Metal backend");
    }

  }
}
