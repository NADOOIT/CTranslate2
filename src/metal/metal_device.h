#pragma once

#include <string>
#include "../device.h"
#include "metal_utils.h"

namespace ctranslate2 {
  namespace metal {

    class MetalDevice : public Device {
    public:
      MetalDevice(int index = 0);
      ~MetalDevice();

      std::string name() const override;
      
      // Device capabilities
      bool has_neural_engine() const;
      
      // Neural Engine configuration
      void set_prefer_neural_engine(bool prefer);
      bool prefers_neural_engine() const;
      
      void synchronize() override;
      void synchronize_stream(void* stream) override;
      void* allocate(std::size_t size) const override;
      void free(void* data) const override;
      void* allocate_stream() const override;
      void free_stream(void* stream) const override;

      // GEMM operations with precision template
      template<typename T>
      void gemm(T* c,
                const T* a,
                const T* b,
                dim_t m,
                dim_t n,
                dim_t k,
                T alpha,
                T beta,
                dim_t lda,
                dim_t ldb,
                dim_t ldc,
                bool transpose_a,
                bool transpose_b,
                void* stream = nullptr) const;

      // Explicit instantiations for float and half precision
      void gemm_fp32(float* c,
                     const float* a,
                     const float* b,
                     dim_t m,
                     dim_t n,
                     dim_t k,
                     float alpha,
                     float beta,
                     dim_t lda,
                     dim_t ldb,
                     dim_t ldc,
                     bool transpose_a,
                     bool transpose_b,
                     void* stream = nullptr) const;

      void gemm_fp16(half* c,
                     const half* a,
                     const half* b,
                     dim_t m,
                     dim_t n,
                     dim_t k,
                     half alpha,
                     half beta,
                     dim_t lda,
                     dim_t ldb,
                     dim_t ldc,
                     bool transpose_a,
                     bool transpose_b,
                     void* stream = nullptr) const;

      // Element-wise operations
      template<typename T>
      void gelu(T* output,
                const T* input,
                dim_t size,
                void* stream = nullptr) const;

      template<typename T>
      void relu(T* output,
                const T* input,
                dim_t size,
                void* stream = nullptr) const;

      template<typename T>
      void add(T* output,
               const T* input1,
               const T* input2,
               dim_t size,
               T alpha = T(1.0),
               T beta = T(1.0),
               void* stream = nullptr) const;

      template<typename T>
      void multiply(T* output,
                    const T* input1,
                    const T* input2,
                    dim_t size,
                    T alpha = T(1.0),
                    T beta = T(0.0),
                    void* stream = nullptr) const;

      // Fused operations for better performance
      template<typename T>
      void gelu_add(T* output,
                    const T* input,
                    const T* residual,
                    dim_t size,
                    void* stream = nullptr) const;

      template<typename T>
      void relu_add(T* output,
                    const T* input,
                    const T* residual,
                    dim_t size,
                    void* stream = nullptr) const;

      template<typename T>
      void layer_norm(T* output,
                     const T* input,
                     const T* gamma,
                     const T* beta,
                     dim_t batch_size,
                     dim_t hidden_size,
                     T epsilon = T(1e-5),
                     void* stream = nullptr) const;

      // Fused activation operations
      template <typename T>
      void fused_gelu(T* output,
                      const T* input,
                      dim_t size,
                      void* stream = nullptr) const;

      template <typename T>
      void fused_relu(T* output,
                      const T* input,
                      dim_t size,
                      void* stream = nullptr) const;

      template <typename T>
      void fused_swish(T* output,
                       const T* input,
                       dim_t size,
                       void* stream = nullptr) const;

      // Dropout operations
      template <typename T>
      void dropout(T* output,
                  const T* input,
                  dim_t size,
                  float dropout_prob,
                  bool scale_output = true,
                  void* stream = nullptr) const;

      template <typename T>
      void dropout_add(T* output,
                      const T* input,
                      const T* residual,
                      dim_t size,
                      float dropout_prob,
                      bool scale_output = true,
                      void* stream = nullptr) const;

      // Attention and softmax operations
      template<typename T>
      void attention(T* output,
                    const T* query,
                    const T* key,
                    const T* value,
                    dim_t batch_size,
                    dim_t num_heads,
                    dim_t head_dim,
                    dim_t seq_len,
                    T scale,
                    void* stream = nullptr) const;

      void softmax(float* output,
                   const float* input,
                   dim_t batch_size,
                   dim_t feature_size,
                   void* stream = nullptr) const;

    private:
      int _device_index;
      MetalUtils& _utils;
      void* _command_queue;
      void* _gemm_pso_fp32;          // Pipeline state object for GEMM (FP32)
      void* _gemm_pso_fp16;          // Pipeline state object for GEMM (FP16)
      void* _relu_pso;          // Pipeline state object for ReLU
      void* _gelu_pso;          // Pipeline state object for GELU
      void* _add_pso;           // Pipeline state object for addition
      void* _multiply_pso;      // Pipeline state object for multiplication
      void* _layer_norm_pso;    // Pipeline state object for layer normalization
      void* _softmax_pso;       // Pipeline state object for softmax
      void* _attention_pso;     // Pipeline state object for attention
      void* _flash_attn_pso;    // Pipeline state object for flash attention
      void* _mps_gemm;  // MPS matrix multiplication primitive
      void* _mps_attention; // MPS attention primitive
      
      // Helper methods
      void* create_pipeline_state(const char* kernel_name) const;
      void* create_mps_kernel(const char* operation) const;
    };

  }  // namespace metal
}  // namespace ctranslate2
