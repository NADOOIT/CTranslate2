#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "metal_kernels.h"
#include "metal_device.h"
#include "ctranslate2/types.h"
#include "ctranslate2/logging.h"

namespace ctranslate2 {
  namespace metal {

    namespace {
      static id<MTLComputePipelineState> get_compute_pipeline(const MetalDevice& device, NSString* kernelName) {
        id<MTLFunction> kernel_function = [device.getLibrary() newFunctionWithName:kernelName];
        if (!kernel_function) {
            throw std::runtime_error("Failed to find the Metal kernel function");
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipelineState = [device.getDevice() newComputePipelineStateWithFunction:kernel_function error:&error];
        if (!pipelineState) {
            throw std::runtime_error("Failed to create compute pipeline state");
        }

        return pipelineState;
      }
    }

    class MetalKernels {
    public:
      void gemm(const MetalDevice& device,
               void* stream,
               bool a_trans,
               bool b_trans,
               dim_t m, dim_t n, dim_t k,
               float alpha,
               const float* a, dim_t lda,
               const float* b, dim_t ldb,
               float beta,
               float* c, dim_t ldc) {
        // Get command buffer from stream
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Set compute pipeline state
        NSString* kernelName = a_trans ? (b_trans ? @"gemm_tt" : @"gemm_tn") 
                                     : (b_trans ? @"gemm_nt" : @"gemm_nn");
        id<MTLComputePipelineState> pipelineState = get_compute_pipeline(device, kernelName);
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers
        void* device_a = device.allocate(m * k * sizeof(float));
        void* device_b = device.allocate(k * n * sizeof(float));
        void* device_c = device.allocate(m * n * sizeof(float));
        
        // Copy data to device buffers
        std::memcpy(device_a, a, m * k * sizeof(float));
        std::memcpy(device_b, b, k * n * sizeof(float));
        std::memcpy(device_c, c, m * n * sizeof(float));
        
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_a offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_b offset:0 atIndex:1];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_c offset:0 atIndex:2];
        
        // Set constants
        struct {
          uint32_t M;
          uint32_t N;
          uint32_t K;
          float alpha;
          float beta;
          uint32_t lda;
          uint32_t ldb;
          uint32_t ldc;
        } constants;
        
        constants.M = static_cast<uint32_t>(m);
        constants.N = static_cast<uint32_t>(n);
        constants.K = static_cast<uint32_t>(k);
        constants.alpha = alpha;
        constants.beta = beta;
        constants.lda = static_cast<uint32_t>(lda);
        constants.ldb = static_cast<uint32_t>(ldb);
        constants.ldc = static_cast<uint32_t>(ldc);
        
        [computeEncoder setBytes:&constants length:sizeof(constants) atIndex:3];
        
        // Dispatch compute kernel
        MTLSize gridSize = MTLSizeMake(m, n, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        // End encoding and commit
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back to host
        std::memcpy(c, device_c, m * n * sizeof(float));
        
        // Free device buffers
        device.free(device_a);
        device.free(device_b);
        device.free(device_c);
      }

      void axpy(const MetalDevice& device,
               void* stream,
               dim_t n,
               float alpha,
               const float* x,
               float* y) {
        // Get command buffer from stream
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Set compute pipeline state
        id<MTLComputePipelineState> pipelineState = get_compute_pipeline(device, @"axpy");
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers
        void* device_x = device.allocate(n * sizeof(float));
        void* device_y = device.allocate(n * sizeof(float));
        
        // Copy data to device buffers
        std::memcpy(device_x, x, n * sizeof(float));
        std::memcpy(device_y, y, n * sizeof(float));
        
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_x offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_y offset:0 atIndex:1];
        
        // Set constants
        struct {
          uint32_t N;
          float alpha;
        } constants;
        
        constants.N = static_cast<uint32_t>(n);
        constants.alpha = alpha;
        
        [computeEncoder setBytes:&constants length:sizeof(constants) atIndex:2];
        
        // Dispatch compute kernel
        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        // End encoding and commit
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back to host
        std::memcpy(y, device_y, n * sizeof(float));
        
        // Free device buffers
        device.free(device_x);
        device.free(device_y);
      }

      void relu(const MetalDevice& device,
               void* stream,
               dim_t size,
               const float* input,
               float* output) {
        // Get command buffer from stream
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Set compute pipeline state
        id<MTLComputePipelineState> pipelineState = get_compute_pipeline(device, @"relu");
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers
        void* device_input = device.allocate(size * sizeof(float));
        void* device_output = device.allocate(size * sizeof(float));
        
        // Copy data to device buffers
        std::memcpy(device_input, input, size * sizeof(float));
        
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_input offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_output offset:0 atIndex:1];
        
        // Set constants
        uint32_t n = static_cast<uint32_t>(size);
        [computeEncoder setBytes:&n length:sizeof(n) atIndex:2];
        
        // Dispatch compute kernel
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        // End encoding and commit
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back to host
        std::memcpy(output, device_output, size * sizeof(float));
        
        // Free device buffers
        device.free(device_input);
        device.free(device_output);
      }

      void layer_norm(const MetalDevice& device,
                     void* stream,
                     dim_t batch_size,
                     dim_t hidden_size,
                     const float* input,
                     const float* gamma,
                     const float* beta,
                     float* output) {
        // Get command buffer from stream
        id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
        
        // Create compute encoder
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        // Set compute pipeline state
        id<MTLComputePipelineState> pipelineState = get_compute_pipeline(device, @"layer_norm");
        [computeEncoder setComputePipelineState:pipelineState];
        
        // Set buffers
        void* device_input = device.allocate(batch_size * hidden_size * sizeof(float));
        void* device_gamma = device.allocate(hidden_size * sizeof(float));
        void* device_beta = device.allocate(hidden_size * sizeof(float));
        void* device_output = device.allocate(batch_size * hidden_size * sizeof(float));
        
        // Copy data to device buffers
        std::memcpy(device_input, input, batch_size * hidden_size * sizeof(float));
        std::memcpy(device_gamma, gamma, hidden_size * sizeof(float));
        std::memcpy(device_beta, beta, hidden_size * sizeof(float));
        
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_input offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_gamma offset:0 atIndex:1];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_beta offset:0 atIndex:2];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)device_output offset:0 atIndex:3];
        
        // Set constants
        struct {
          uint32_t batch_size;
          uint32_t hidden_size;
        } constants;
        
        constants.batch_size = static_cast<uint32_t>(batch_size);
        constants.hidden_size = static_cast<uint32_t>(hidden_size);
        
        [computeEncoder setBytes:&constants length:sizeof(constants) atIndex:4];
        
        // Dispatch compute kernel
        MTLSize gridSize = MTLSizeMake(batch_size * hidden_size, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        // End encoding and commit
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back to host
        std::memcpy(output, device_output, batch_size * hidden_size * sizeof(float));
        
        // Free device buffers
        device.free(device_input);
        device.free(device_gamma);
        device.free(device_beta);
        device.free(device_output);
      }
    };

    void add(MetalDevice& device,
             const float* a,
             const float* b,
             float* c,
             std::size_t size,
             void* stream) {
      MetalKernels kernels;
      kernels.axpy(device, stream, size, 1.0f, a, c);
      kernels.axpy(device, stream, size, 1.0f, b, c);
    }

    void multiply(MetalDevice& device,
                 const float* a,
                 const float* b,
                 float* c,
                 std::size_t size,
                 void* stream) {
      MetalKernels kernels;
      kernels.axpy(device, stream, size, 1.0f, a, c);
      kernels.axpy(device, stream, size, 1.0f, b, c);
    }

    void softmax(MetalDevice& device,
                const float* input,
                float* output,
                std::size_t batch_size,
                std::size_t depth,
                void* stream) {
      // Not implemented
    }

    void layer_norm(MetalDevice& device,
                   const float* input,
                   const float* gamma,
                   const float* beta,
                   float* output,
                   std::size_t batch_size,
                   std::size_t hidden_size,
                   float epsilon,
                   void* stream) {
      MetalKernels kernels;
      kernels.layer_norm(device, stream, batch_size, hidden_size, input, gamma, beta, output);
    }

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
              void* stream) {
      MetalKernels kernels;
      kernels.gemm(device, stream, a_trans, b_trans, m, n, k, alpha, a, m, b, k, beta, c, n);
    }

  }  // namespace metal
}  // namespace ctranslate2
