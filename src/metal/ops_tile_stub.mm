#include "ctranslate2/ops/tile.h"

#include <stdexcept>

#include "metal_utils.h"
#include "ctranslate2/types.h"
#include <Eigen/Core>
#include <half_float/half.hpp>

namespace ctranslate2 {
  namespace ops {

    template<>
    void Tile::compute<Device::METAL, float>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (float) is not implemented for Metal backend");
    }

    template<>
    void Tile::compute<Device::METAL, half_float::half>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (half_float::half) is not implemented for Metal backend");
    }

    template<>
    void Tile::compute<Device::METAL, bfloat16_t>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (bfloat16_t) is not implemented for Metal backend");
    }

    template<>
    void Tile::compute<Device::METAL, int8_t>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (int8_t) is not implemented for Metal backend");
    }

    template<>
    void Tile::compute<Device::METAL, int16_t>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (int16_t) is not implemented for Metal backend");
    }

    template<>
    void Tile::compute<Device::METAL, int32_t>(
        const StorageView& input,
        const dim_t outer_size,
        const dim_t inner_size,
        StorageView& output) const {
      throw std::runtime_error("Tile Op (int32_t) is not implemented for Metal backend");
    }

  }
}
