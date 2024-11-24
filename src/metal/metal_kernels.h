#pragma once

#include <cstddef>
#include <cstdint>

namespace ctranslate2 {
  namespace metal {

    // Forward declarations
    class MetalDevice;

    // Basic compute operations
    void add(MetalDevice& device,
             const float* a,
             const float* b,
             float* c,
             std::size_t size,
             void* stream = nullptr);

    void multiply(MetalDevice& device,
                 const float* a,
                 const float* b,
                 float* c,
                 std::size_t size,
                 void* stream = nullptr);

    void relu(MetalDevice& device,
              const float* x,
              float* y,
              std::size_t size,
              void* stream = nullptr);

    void softmax(MetalDevice& device,
                const float* input,
                float* output,
                std::size_t batch_size,
                std::size_t depth,
                void* stream = nullptr);

    void layer_norm(MetalDevice& device,
                   const float* input,
                   const float* gamma,
                   const float* beta,
                   float* output,
                   std::size_t batch_size,
                   std::size_t hidden_size,
                   float epsilon = 1e-6f,
                   void* stream = nullptr);

    // Matrix operations
    void gemm(MetalDevice& device,
              bool a_trans,
              bool b_trans,
              std::size_t m,
              std::size_t n,
              std::size_t k,
              float alpha,
              const float* a,
              const float* b,
              float beta,
              float* c,
              void* stream = nullptr);

  }  // namespace metal
}  // namespace ctranslate2
