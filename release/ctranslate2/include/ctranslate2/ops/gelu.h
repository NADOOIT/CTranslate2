#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class GELU : public UnaryOp {
    public:
      enum class Approximation {
        None,
        Tanh,
        Sigmoid,
      };

      GELU(const Approximation approximation = Approximation::None);

      void operator()(const StorageView& x, StorageView& y) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& x, StorageView& y) const;

      const Approximation _approximation;
    };

  }
}
