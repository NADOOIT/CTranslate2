#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_device.h"
#include <sstream>
#include <stdexcept>

namespace ctranslate2 {
  namespace metal {

    MetalDevice::MetalDevice(bool prefer_neural_engine)
      : _prefer_neural_engine(prefer_neural_engine) {
      _metal_device = MTLCreateSystemDefaultDevice();
      if (!_metal_device) {
        throw std::runtime_error("No Metal device found");
      }
      create_command_queue();
      create_pipeline_states();
    }

    MetalDevice::~MetalDevice() {
    }

    std::string MetalDevice::name() const {
      NSString* device_name = [_metal_device name];
      return std::string([device_name UTF8String]);
    }

    void MetalDevice::synchronize() {
      if (_command_queue) {
        id<MTLCommandBuffer> command_buffer = [_command_queue commandBuffer];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
      }
    }

    void MetalDevice::synchronize_stream(void* stream) {
      if (stream) {
        id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)stream;
        [command_buffer waitUntilCompleted];
      }
    }

    void* MetalDevice::allocate(std::size_t size) const {
      id<MTLBuffer> buffer = [_metal_device newBufferWithLength:size
                                                      options:MTLResourceStorageModeShared];
      if (!buffer) {
        throw std::runtime_error("Failed to allocate Metal buffer");
      }
      return (__bridge_retained void*)buffer;
    }

    void MetalDevice::free(void* data) const {
      if (data) {
        CFRelease(data);
      }
    }

    void* MetalDevice::allocate_stream() const {
      id<MTLCommandBuffer> command_buffer = [_command_queue commandBuffer];
      return (__bridge_retained void*)command_buffer;
    }

    void MetalDevice::free_stream(void* stream) const {
      if (stream) {
        CFRelease(stream);
      }
    }

    void MetalDevice::create_command_queue() {
      _command_queue = [_metal_device newCommandQueue];
      if (!_command_queue) {
        throw std::runtime_error("Failed to create Metal command queue");
      }
    }

    void MetalDevice::create_pipeline_states() {
      NSError* error = nil;
      
      // Load default library
      _library = [_metal_device newDefaultLibrary];
      if (!_library) {
        throw std::runtime_error("Failed to create Metal library");
      }

      // Create GEMM pipeline state
      id<MTLFunction> gemm_function = [_library newFunctionWithName:@"gemm"];
      if (!gemm_function) {
        throw std::runtime_error("Failed to find GEMM kernel");
      }
      _gemm_pipeline = [_metal_device newComputePipelineStateWithFunction:gemm_function error:&error];
      if (!_gemm_pipeline) {
        throw std::runtime_error("Failed to create GEMM pipeline state");
      }

      // Create ReLU pipeline state
      id<MTLFunction> relu_function = [_library newFunctionWithName:@"relu"];
      if (!relu_function) {
        throw std::runtime_error("Failed to find ReLU kernel");
      }
      _relu_pipeline = [_metal_device newComputePipelineStateWithFunction:relu_function error:&error];
      if (!_relu_pipeline) {
        throw std::runtime_error("Failed to create ReLU pipeline state");
      }

      // Create GELU pipeline state
      id<MTLFunction> gelu_function = [_library newFunctionWithName:@"gelu"];
      if (!gelu_function) {
        throw std::runtime_error("Failed to find GELU kernel");
      }
      _gelu_pipeline = [_metal_device newComputePipelineStateWithFunction:gelu_function error:&error];
      if (!_gelu_pipeline) {
        throw std::runtime_error("Failed to create GELU pipeline state");
      }

      // Create Add pipeline state
      id<MTLFunction> add_function = [_library newFunctionWithName:@"add"];
      if (!add_function) {
        throw std::runtime_error("Failed to find Add kernel");
      }
      _add_pipeline = [_metal_device newComputePipelineStateWithFunction:add_function error:&error];
      if (!_add_pipeline) {
        throw std::runtime_error("Failed to create Add pipeline state");
      }

      // Create Multiply pipeline state
      id<MTLFunction> multiply_function = [_library newFunctionWithName:@"multiply"];
      if (!multiply_function) {
        throw std::runtime_error("Failed to find Multiply kernel");
      }
      _multiply_pipeline = [_metal_device newComputePipelineStateWithFunction:multiply_function error:&error];
      if (!_multiply_pipeline) {
        throw std::runtime_error("Failed to create Multiply pipeline state");
      }

      // Create Softmax pipeline state
      id<MTLFunction> softmax_function = [_library newFunctionWithName:@"softmax"];
      if (!softmax_function) {
        throw std::runtime_error("Failed to find Softmax kernel");
      }
      _softmax_pipeline = [_metal_device newComputePipelineStateWithFunction:softmax_function error:&error];
      if (!_softmax_pipeline) {
        throw std::runtime_error("Failed to create Softmax pipeline state");
      }

      // Create Attention pipeline state
      id<MTLFunction> attention_function = [_library newFunctionWithName:@"attention"];
      if (!attention_function) {
        throw std::runtime_error("Failed to find Attention kernel");
      }
      _attention_pipeline = [_metal_device newComputePipelineStateWithFunction:attention_function error:&error];
      if (!_attention_pipeline) {
        throw std::runtime_error("Failed to create Attention pipeline state");
      }
    }

    void MetalDevice::gemm(float* c,
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
                          void* stream) const {
      // Create Metal buffers
      id<MTLBuffer> a_buffer = [_metal_device newBufferWithBytes:a length:m * k * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> b_buffer = [_metal_device newBufferWithBytes:b length:k * n * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> c_buffer = [_metal_device newBufferWithBytes:c length:m * n * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_gemm_pipeline];
      [computeEncoder setBuffer:a_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:b_buffer offset:0 atIndex:1];
      [computeEncoder setBuffer:c_buffer offset:0 atIndex:2];
      [computeEncoder setBytes:&m length:sizeof(dim_t) atIndex:3];
      [computeEncoder setBytes:&n length:sizeof(dim_t) atIndex:4];
      [computeEncoder setBytes:&k length:sizeof(dim_t) atIndex:5];
      [computeEncoder setBytes:&alpha length:sizeof(float) atIndex:6];
      [computeEncoder setBytes:&beta length:sizeof(float) atIndex:7];
      [computeEncoder setBytes:&transpose_a length:sizeof(bool) atIndex:8];
      [computeEncoder setBytes:&transpose_b length:sizeof(bool) atIndex:9];

      // Calculate grid and threadgroup sizes
      MTLSize grid_size = MTLSizeMake(m, n, 1);
      MTLSize threadgroup_size = MTLSizeMake(16, 16, 1);  // Adjust based on device capabilities
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(c, [c_buffer contents], m * n * sizeof(float));
    }

    void MetalDevice::relu(float* output,
                          const float* input,
                          dim_t size,
                          void* stream) const {
      // Create Metal buffers
      id<MTLBuffer> input_buffer = [_metal_device newBufferWithBytes:input length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:size * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_relu_pipeline];
      [computeEncoder setBuffer:input_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:1];
      [computeEncoder setBytes:&size length:sizeof(dim_t) atIndex:2];

      // Calculate grid and threadgroup sizes
      NSUInteger thread_group_size = _relu_pipeline.maxTotalThreadsPerThreadgroup;
      MTLSize grid_size = MTLSizeMake(size, 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], size * sizeof(float));
    }

    void MetalDevice::gelu(float* output,
                          const float* input,
                          dim_t size,
                          void* stream) const {
      // Create Metal buffers
      id<MTLBuffer> input_buffer = [_metal_device newBufferWithBytes:input length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:size * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_gelu_pipeline];
      [computeEncoder setBuffer:input_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:1];
      [computeEncoder setBytes:&size length:sizeof(dim_t) atIndex:2];

      // Calculate grid and threadgroup sizes
      NSUInteger thread_group_size = _gelu_pipeline.maxTotalThreadsPerThreadgroup;
      MTLSize grid_size = MTLSizeMake(size, 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], size * sizeof(float));
    }

    void MetalDevice::add(float* output,
                         const float* input1,
                         const float* input2,
                         dim_t size,
                         float alpha,
                         float beta,
                         void* stream) const {
      // Create Metal buffers
      id<MTLBuffer> input1_buffer = [_metal_device newBufferWithBytes:input1 length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> input2_buffer = [_metal_device newBufferWithBytes:input2 length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:size * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_add_pipeline];
      [computeEncoder setBuffer:input1_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:input2_buffer offset:0 atIndex:1];
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:2];
      [computeEncoder setBytes:&size length:sizeof(dim_t) atIndex:3];
      [computeEncoder setBytes:&alpha length:sizeof(float) atIndex:4];
      [computeEncoder setBytes:&beta length:sizeof(float) atIndex:5];

      // Calculate grid and threadgroup sizes
      NSUInteger thread_group_size = _add_pipeline.maxTotalThreadsPerThreadgroup;
      MTLSize grid_size = MTLSizeMake(size, 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], size * sizeof(float));
    }

    void MetalDevice::multiply(float* output,
                             const float* input1,
                             const float* input2,
                             dim_t size,
                             float alpha,
                             float beta,
                             void* stream) const {
      // Create Metal buffers
      id<MTLBuffer> input1_buffer = [_metal_device newBufferWithBytes:input1 length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> input2_buffer = [_metal_device newBufferWithBytes:input2 length:size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:size * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_multiply_pipeline];
      [computeEncoder setBuffer:input1_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:input2_buffer offset:0 atIndex:1];
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:2];
      [computeEncoder setBytes:&size length:sizeof(dim_t) atIndex:3];
      [computeEncoder setBytes:&alpha length:sizeof(float) atIndex:4];
      [computeEncoder setBytes:&beta length:sizeof(float) atIndex:5];

      // Calculate grid and threadgroup sizes
      NSUInteger thread_group_size = _multiply_pipeline.maxTotalThreadsPerThreadgroup;
      MTLSize grid_size = MTLSizeMake(size, 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], size * sizeof(float));
    }

    void MetalDevice::softmax(float* output,
                            const float* input,
                            dim_t batch_size,
                            dim_t feature_size,
                            void* stream) const {
      // Create Metal buffers
      size_t total_size = batch_size * feature_size;
      id<MTLBuffer> input_buffer = [_metal_device newBufferWithBytes:input length:total_size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:total_size * sizeof(float) options:MTLResourceStorageModeShared];

      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_softmax_pipeline];
      [computeEncoder setBuffer:input_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:1];
      [computeEncoder setBytes:&batch_size length:sizeof(dim_t) atIndex:2];
      [computeEncoder setBytes:&feature_size length:sizeof(dim_t) atIndex:3];

      // Calculate grid and threadgroup sizes
      MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
      MTLSize threadgroup_size = MTLSizeMake(std::min(256u, (unsigned int)batch_size), 1, 1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], total_size * sizeof(float));
    }

    void MetalDevice::attention(float* output,
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
                               void* stream) const {
      // Calculate sizes
      size_t qk_size = batch_size * num_heads * queries_per_head * keys_per_head;
      size_t q_size = batch_size * num_heads * queries_per_head * dim_per_head;
      size_t k_size = batch_size * num_heads * keys_per_head * dim_per_head;
      size_t v_size = k_size;
      size_t output_size = q_size;
      
      // Create Metal buffers
      id<MTLBuffer> query_buffer = [_metal_device newBufferWithBytes:query length:q_size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> key_buffer = [_metal_device newBufferWithBytes:key length:k_size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> value_buffer = [_metal_device newBufferWithBytes:value length:v_size * sizeof(float) options:MTLResourceStorageModeShared];
      id<MTLBuffer> mask_buffer = mask ? [_metal_device newBufferWithBytes:mask length:qk_size * sizeof(float) options:MTLResourceStorageModeShared] : nil;
      id<MTLBuffer> output_buffer = [_metal_device newBufferWithBytes:output length:output_size * sizeof(float) options:MTLResourceStorageModeShared];
      
      // Create command buffer and encoder
      id<MTLCommandBuffer> commandBuffer = [_command_queue commandBuffer];
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      
      // Set pipeline state and arguments
      [computeEncoder setComputePipelineState:_attention_pipeline];
      [computeEncoder setBuffer:query_buffer offset:0 atIndex:0];
      [computeEncoder setBuffer:key_buffer offset:0 atIndex:1];
      [computeEncoder setBuffer:value_buffer offset:0 atIndex:2];
      if (mask_buffer) {
          [computeEncoder setBuffer:mask_buffer offset:0 atIndex:3];
      }
      [computeEncoder setBuffer:output_buffer offset:0 atIndex:4];
      
      // Set attention parameters
      struct {
          uint32_t batch_size;
          uint32_t num_heads;
          uint32_t queries_per_head;
          uint32_t keys_per_head;
          uint32_t dim_per_head;
          float scale;
          uint32_t is_causal;
      } params = {
          static_cast<uint32_t>(batch_size),
          static_cast<uint32_t>(num_heads),
          static_cast<uint32_t>(queries_per_head),
          static_cast<uint32_t>(keys_per_head),
          static_cast<uint32_t>(dim_per_head),
          scale,
          static_cast<uint32_t>(is_causal)
      };
      [computeEncoder setBytes:&params length:sizeof(params) atIndex:5];
      
      // Calculate grid and threadgroup sizes
      // Each thread processes one query position for all heads
      MTLSize grid_size = MTLSizeMake(queries_per_head, batch_size, 1);
      MTLSize threadgroup_size = MTLSizeMake(std::min(32u, (unsigned int)queries_per_head),
                                           std::min(32u, (unsigned int)batch_size),
                                           1);
      [computeEncoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
      
      // End encoding and commit
      [computeEncoder endEncoding];
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];

      // Copy result back to host
      memcpy(output, [output_buffer contents], output_size * sizeof(float));
    }

  }  // namespace metal
}  // namespace ctranslate2
