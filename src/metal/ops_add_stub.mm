// src/metal/ops_add_stub.mm
#include "ctranslate2/ops/add.h"
#include "ctranslate2/utils.h" // For bfloat16_t, etc.
#include "dispatch.h"          // For Device enum
#include <Eigen/Core>          // For Eigen::half
#include <stdexcept>

namespace ctranslate2 {
  namespace ops {

    template<>
    void Add::compute<Device::METAL, float>(
        const StorageView& a,
        const StorageView& b,
        StorageView& c) const {
      (void)a; (void)b; (void)c;
      throw std::runtime_error("Add::compute for Device::METAL (float) not implemented yet");
    }

    template<>
    void Add::compute<Device::METAL, Eigen::half>(
        const StorageView& a,
        const StorageView& b,
        StorageView& c) const {
      (void)a; (void)b; (void)c;
      throw std::runtime_error("Add::compute for Device::METAL (Eigen::half) not implemented yet");
    }

    template<>
    void Add::compute<Device::METAL, bfloat16_t>(
        const StorageView& a,
        const StorageView& b,
        StorageView& c) const {
      (void)a; (void)b; (void)c;
      throw std::runtime_error("Add::compute for Device::METAL (bfloat16_t) not implemented yet");
    }

    template<>
    void Add::compute<Device::METAL, int8_t>(
        const StorageView& a,
        const StorageView& b,
        StorageView& c) const {
      (void)a; (void)b; (void)c;
      throw std::runtime_error("Add::compute for Device::METAL (int8_t) not implemented yet");
    }

    // TODO: Add specializations for other types if linker errors show them

  }
}
