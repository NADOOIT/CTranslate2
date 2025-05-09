#include "ctranslate2/ops/gather.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h"
#include <Eigen/Core> // For Eigen::half
#include <half_float/half.hpp> // For half_float::half

namespace ctranslate2 {
  namespace ops {

    template<>
    void Gather::compute<Device::METAL, float>(const StorageView& data,
                                               const StorageView& input,
                                               const dim_t axis,
                                               const dim_t batch_dims,
                                               StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (float)");
    }

#ifdef CT2_WITH_HALF
    template<>
    void Gather::compute<Device::METAL, Eigen::half>(const StorageView& data,
                                                     const StorageView& input,
                                                     const dim_t axis,
                                                     const dim_t batch_dims,
                                                     StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (Eigen::half)");
    }
#endif

    template<>
    void Gather::compute<Device::METAL, half_float::half>(const StorageView& data,
                                                          const StorageView& input,
                                                          const dim_t axis,
                                                          const dim_t batch_dims,
                                                          StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (half_float::half)");
    }

#ifdef CT2_WITH_BFLOAT16
    template<>
    void Gather::compute<Device::METAL, bfloat16_t>(const StorageView& data,
                                                    const StorageView& input,
                                                    const dim_t axis,
                                                    const dim_t batch_dims,
                                                    StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (bfloat16_t)");
    }
#endif

    template<>
    void Gather::compute<Device::METAL, bfloat16_t>(const StorageView& data,
                                                    const StorageView& input,
                                                    const dim_t axis,
                                                    const dim_t batch_dims,
                                                    StorageView& output) const {
      // This is the unconditional bfloat16_t from the linker error
      throw std::runtime_error("Gather Op is not implemented for Metal backend (unconditional bfloat16_t)");
    }

    template<>
    void Gather::compute<Device::METAL, int8_t>(const StorageView& data,
                                                const StorageView& input,
                                                const dim_t axis,
                                                const dim_t batch_dims,
                                                StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (int8_t)");
    }

    template<>
    void Gather::compute<Device::METAL, int16_t>(const StorageView& data,
                                                 const StorageView& input,
                                                 const dim_t axis,
                                                 const dim_t batch_dims,
                                                 StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (int16_t)");
    }

    template<>
    void Gather::compute<Device::METAL, int32_t>(const StorageView& data,
                                                 const StorageView& input,
                                                 const dim_t axis,
                                                 const dim_t batch_dims,
                                                 StorageView& output) const {
      throw std::runtime_error("Gather Op is not implemented for Metal backend (int32_t)");
    }

  }
}
