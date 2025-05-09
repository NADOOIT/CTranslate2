// This file contains the Metal stub for LayerNorm::compute.
// It is intended to be included directly by src/ops/layer_norm.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/layer_norm.h" // For LayerNorm class definition
#include "ctranslate2/devices.h"      // For Device::METAL
#include "ctranslate2/storage_view.h"

#include <stdexcept> // For std::runtime_error

namespace ctranslate2 {
  namespace ops {

    // Metal specialization for LayerNorm::compute
    template<>
    void LayerNorm::compute<Device::METAL, float>(
        const StorageView* beta,
        const StorageView* gamma,
        const StorageView& input,
        dim_t axis,
        dim_t outer_size,
        dim_t axis_size,
        dim_t inner_size,
        StorageView& output) const {
      // TODO: Implement Metal LayerNorm
      (void)beta; (void)gamma; (void)input; (void)axis; (void)outer_size; (void)axis_size; (void)inner_size; (void)output;
      throw std::runtime_error("LayerNorm for Device::METAL (float) not implemented yet - included stub");
    }

  } // namespace ops
} // namespace ctranslate2
