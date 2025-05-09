#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class Add : public BinaryOp {
    public:
      void operator()(const StorageView& a, const StorageView& b, StorageView& c) const override;

    private:
      template <Device D, typename T>
      void compute(const StorageView& a, const StorageView& b, StorageView& c) const;
    };

  }
}
