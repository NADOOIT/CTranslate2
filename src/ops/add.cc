#include "ctranslate2/ops/add.h"

#include "dispatch.h"

// #ifdef CT2_WITH_METAL
// #  include "../metal/ops_add_stub.mm" // Removed: stub is compiled separately
// #endif

namespace ctranslate2 {
  namespace ops {

    // Primary template definition for Add::compute
    template <Device D, typename T>
    void Add::compute(const StorageView& a, const StorageView& b, StorageView& c) const {
      c.resize_as(a);
      if (b.is_scalar()) {
        primitives<D>::add(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());
      } else {
        primitives<D>::add(a.data<T>(), b.data<T>(), c.data<T>(), c.size());
      }
    }

    void Add::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("Add");
      DEVICE_AND_TYPE_DISPATCH(a.device(), a.dtype(), (compute<D, T>(a, b, c)));
    }

  }
}
