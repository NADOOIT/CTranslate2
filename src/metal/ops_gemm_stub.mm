// This file contains the Metal stub for GEMM::compute.
// It is intended to be included directly by src/ops/gemm.cc when CT2_WITH_METAL is defined.

#include "ctranslate2/ops/gemm.h" // For GEMM class definition (still needed for full def)
#include "ctranslate2/devices.h"  // For Device::METAL
#include "ctranslate2/storage_view.h"
#include <stdexcept> // For std::runtime_error

// Forward declarations
namespace ctranslate2 {
  namespace ops {
    class Gemm; // Forward declaration of the class itself

    // Forward declaration of the specific member template specialization
    template<>
    void Gemm::compute<Device::METAL, float, float>(
        const StorageView& a,
        const StorageView& b,
        const StorageView* a_shift_compensation, // Swapped
        StorageView& c) const;                   // Swapped
  }
}

// Definition of the specialization
namespace ctranslate2 {
  namespace ops {

    template<>
    void Gemm::compute<Device::METAL, float, float>(
        const StorageView& a,
        const StorageView& b,
        const StorageView* a_shift_compensation, // Swapped
        StorageView& c) const {                 // Swapped
      // TODO: Implement Metal GEMM
      (void)a; (void)b; (void)c; (void)a_shift_compensation;
      throw std::runtime_error("GEMM for Device::METAL (float,float) not implemented yet - stub with fwd decl");
    }

  } // namespace ops
} // namespace ctranslate2
