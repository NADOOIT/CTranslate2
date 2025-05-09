// include/ctranslate2/ops/move.h
#pragma once
#include "ctranslate2/storage_view.h" // For StorageView, Device
#include "ctranslate2/devices.h"    // For Device enum

namespace ctranslate2 {
  namespace ops {

    class Move {
    public:
      // This is the primary template that will be specialized.
      template <Device D, typename T>
      static void compute(const StorageView& input, StorageView& output);
    };

    // Definition for non-specialized cases (e.g., CPU default if any)
    // Or leave it undefined to force specialization.
    // For now, leave it undefined as we only care about Metal.

  }
}
