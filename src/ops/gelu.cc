#include "ctranslate2/ops/gelu.h" // Includes op.h -> primitives.h

#include "dispatch.h"
// cpu/kernels.h is not directly needed here as primitives<D> handles it.

#ifdef CT2_WITH_METAL
#  include "../metal/ops_gelu_stub.mm" // Must be before first instantiation
#endif

namespace ctranslate2 {
  namespace ops {

    // Primary template definition for GELU::compute
    template <Device D, typename T>
    void GELU::compute(const StorageView& x, StorageView& y) const {
      // This is the generic implementation, relying on primitives<D>
      // for device-specific dispatch (including CPU ISA dispatch).
      switch (_approximation) {
      case Approximation::None:
        primitives<D>::gelu(x.data<T>(), y.data<T>(), x.size());
        break;
      case Approximation::Tanh:
        primitives<D>::gelu_tanh(x.data<T>(), y.data<T>(), x.size());
        break;
      case Approximation::Sigmoid:
        primitives<D>::gelu_sigmoid(x.data<T>(), y.data<T>(), x.size());
        break;
      }
    }

    GELU::GELU(const Approximation approximation)
      : _approximation(approximation)
    {
    }

    void GELU::operator()(const StorageView& x, StorageView& y) const {
      PROFILE("GELU");

      y.resize_as(x);

      DEVICE_AND_FLOAT_DISPATCH("GELU", x.device(), x.dtype(), (compute<D, T>(x, y)));
    }

  }
}
