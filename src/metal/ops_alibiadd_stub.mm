// This file contains the Metal stub for AlibiAdd::compute.
// It is intended to be included directly by src/ops/alibi_add.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/alibi_add.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/storage_view.h"
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {
    template<>
    void AlibiAdd::compute<Device::METAL, float>(
        const StorageView& alibi,
        const StorageView& input,
        dim_t layer_idx,
        StorageView& output) const {
      (void)alibi; (void)input; (void)layer_idx; (void)output;
      throw std::runtime_error("AlibiAdd for Device::METAL (float) not implemented yet - included stub");
    }
  }
}
