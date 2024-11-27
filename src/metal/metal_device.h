#pragma once

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "../device.h"
#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace metal {

    class MetalDevice : public IDevice {
    public:
      MetalDevice(bool prefer_neural_engine = true);
      ~MetalDevice() override;

      std::string name() const override;
      void synchronize() override;
      void synchronize_stream(void* stream) override;
      void* allocate(std::size_t size) const override;
      void free(void* data) const override;
      void* allocate_stream() const override;
      void free_stream(void* stream) const override;

      // Metal-specific operations
      void gemm(float* c,
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

      void relu(float* output,
                const float* input,
                dim_t size,
                void* stream = nullptr) const;

      void gelu(float* output,
                const float* input,
                dim_t size,
                void* stream = nullptr) const;

      void add(float* output,
               const float* input1,
               const float* input2,
               dim_t size,
               float alpha = 1.0f,
               float beta = 1.0f,
               void* stream = nullptr) const;

      void multiply(float* output,
                    const float* input1,
                    const float* input2,
                    dim_t size,
                    float alpha = 1.0f,
                    float beta = 0.0f,
                    void* stream = nullptr) const;

      void softmax(float* output,
                   const float* input,
                   dim_t batch_size,
                   dim_t feature_size,
                   void* stream = nullptr) const;

      void attention(float* output,
                    const float* query,
                    const float* key,
                    const float* value,
                    const float* mask,
                    dim_t batch_size,
                    dim_t num_heads,
                    dim_t queries_per_head,
                    dim_t keys_per_head,
                    dim_t dim_per_head,
                    float scale,
                    bool is_causal,
                    void* stream = nullptr) const;

      // Metal utilities
      id<MTLDevice> getDevice() const { return _metal_device; }
      id<MTLCommandQueue> getCommandQueue() const { return _command_queue; }
      id<MTLLibrary> getLibrary() const { return _library; }

    private:
      void create_command_queue();
      void create_pipeline_states();

      id<MTLDevice> _metal_device;
      id<MTLCommandQueue> _command_queue;
      id<MTLLibrary> _library;
      id<MTLComputePipelineState> _gemm_pipeline;
      id<MTLComputePipelineState> _relu_pipeline;
      id<MTLComputePipelineState> _gelu_pipeline;
      id<MTLComputePipelineState> _add_pipeline;
      id<MTLComputePipelineState> _multiply_pipeline;
      id<MTLComputePipelineState> _softmax_pipeline;
      id<MTLComputePipelineState> _attention_pipeline;
      bool _prefer_neural_engine;
    };

  } // namespace metal
} // namespace ctranslate2
