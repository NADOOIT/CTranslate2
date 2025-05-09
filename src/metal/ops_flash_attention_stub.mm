#include "ctranslate2/ops/flash_attention.h"

#include <stdexcept>

#include "metal_utils.h"

namespace ctranslate2 {
  namespace ops {

    template<>
    void FlashAttention::compute<Device::METAL>(
        StorageView& queries,
        StorageView& keys,
        StorageView& values,
        StorageView& output,
        StorageView* cached_keys,
        StorageView* cached_values,
        StorageView* attention,
        bool return_normalized_attention,
        StorageView* rotary_cos,
        StorageView* rotary_sin,
        const bool rotary_interleave,
        StorageView* alibi,
        dim_t offset) const {
      throw std::runtime_error("FlashAttention Op is not implemented for Metal backend");
    }

  }
}
