#pragma once

#include "op.h"

namespace ctranslate2 {
  namespace ops {

    class LayerNorm : public TernaryOp {
    public:
      LayerNorm(const dim_t axis = -1, const float epsilon = 1e-5);

      // This operator implements the TernaryOp interface.
      void operator()(const StorageView& beta,
                      const StorageView& gamma,
                      const StorageView& input,
                      StorageView& output) const override;

      // Other public operator overloads specific to LayerNorm
      void operator()(StorageView& input) const;
      void operator()(const StorageView& input, StorageView& output) const;

    private:
      void operator()(const StorageView* beta,
                      const StorageView* gamma,
                      const StorageView& input,
                      StorageView& output) const;

      template <Device D, typename T>
      void compute(const StorageView* beta,
                   const StorageView* gamma,
                   const StorageView& input,
                   dim_t axis,
                   dim_t outer_size,
                   dim_t axis_size,
                   dim_t inner_size,
                   StorageView& output) const;

      const dim_t _axis;
      const float _epsilon;
    };

  } // namespace ops
} // namespace ctranslate2
