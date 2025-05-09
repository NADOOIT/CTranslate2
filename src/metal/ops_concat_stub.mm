#include "ctranslate2/ops/concat.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h"
#include <Eigen/Core> // For Eigen::half
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void Concat::compute<Device::METAL, float>(const std::vector<const StorageView*>& inputs,
                                               StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (float)");
    }

#ifdef CT2_WITH_HALF
    template<>
    void Concat::compute<Device::METAL, Eigen::half>(const std::vector<const StorageView*>& inputs,
                                                     StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (Eigen::half)");
    }
#endif

    template<>
    void Concat::compute<Device::METAL, half_float::half>(const std::vector<const StorageView*>& inputs,
                                                          StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (half_float::half)");
    }

#ifdef CT2_WITH_BFLOAT16
    template<>
    void Concat::compute<Device::METAL, bfloat16_t>(const std::vector<const StorageView*>& inputs,
                                                    StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (bfloat16_t)");
    }
#endif

    template<>
    void Concat::compute<Device::METAL, bfloat16_t>(const std::vector<const StorageView*>& inputs,
                                                    StorageView& output) const {
      // This is the unconditional bfloat16_t from the linker error
      throw std::runtime_error("Concat Op is not implemented for Metal backend (unconditional bfloat16_t)");
    }

    template<>
    void Concat::compute<Device::METAL, int8_t>(const std::vector<const StorageView*>& inputs,
                                                StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (int8_t)");
    }

    template<>
    void Concat::compute<Device::METAL, int16_t>(const std::vector<const StorageView*>& inputs,
                                                 StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (int16_t)");
    }

    template<>
    void Concat::compute<Device::METAL, int32_t>(const std::vector<const StorageView*>& inputs,
                                                 StorageView& output) const {
      throw std::runtime_error("Concat Op is not implemented for Metal backend (int32_t)");
    }

    // Add other types if needed based on linker errors.

  }
}
