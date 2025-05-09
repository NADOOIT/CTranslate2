// src/metal/ops_move_stub.mm
#include "ctranslate2/ops/move.h" // Includes the definition of ops::Move
#include <stdexcept>              // For runtime_error

namespace ctranslate2 {
  namespace ops {

    // Definition of the template specialization for the static member Move::compute
    template<>
    void Move::compute<Device::METAL, float>(
        const StorageView& input,
        StorageView& output) { // No 'const' as it's a static member
      (void)input; (void)output;
      // This is usually a device-to-device copy.
      // For Metal, this would involve creating a command buffer, a blit encoder,
      // and dispatching a copy command if input and output are on Device::METAL.
      // If one is CPU, it would be a CPU<->Metal copy.
      throw std::runtime_error("Move::compute for Device::METAL (float) not implemented yet - defined in ops_move_stub.mm, using ops/move.h");
    }

  }
}
