#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "metal_device.h"
#include "ctranslate2/logging.h"

namespace ctranslate2 {
  namespace metal {

    MetalDevice::MetalDevice(int index)
      : _device_index(index)
      , _utils(MetalUtils::getInstance())
      , _command_queue(nullptr)
      , _gemm_pso_fp32(nullptr)
      , _gemm_pso_fp16(nullptr)
      , _relu_pso(nullptr)
      , _gelu_pso(nullptr)
      , _add_pso(nullptr)
      , _multiply_pso(nullptr)
      , _layer_norm_pso(nullptr)
      , _softmax_pso(nullptr)
      , _attention_pso(nullptr)
      , _flash_attn_pso(nullptr)
      , _prefer_neural_engine(true)  // Default to preferring Neural Engine when available
      , _gemm_mlmodel(nullptr)
      , _attention_mlmodel(nullptr)
      , _mps_gemm(nullptr)
      , _mps_attention(nullptr) {
      
      @autoreleasepool {
        // Get device and command queue from MetalUtils
        id<MTLDevice> device = _utils.getDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
          throw std::runtime_error("Failed to create Metal command queue");
        }
        
        _command_queue = (__bridge_retained void*)queue;
        
        // Create MPS matrix multiplication primitive
        MPSMatrixMultiplication* mpsGEMM = [[MPSMatrixMultiplication alloc] initWithDevice:device];
        _mps_gemm = (__bridge_retained void*)mpsGEMM;
        
        // Create MPS attention primitive
        if (@available(macOS 13.0, *)) {
          MPSGraphMultiheadAttention* mpsAttention = [[MPSGraphMultiheadAttention alloc] initWithDevice:device];
          _mps_attention = (__bridge_retained void*)mpsAttention;
        }
      }
    }

    MetalDevice::~MetalDevice() {
      if (_command_queue) {
        CFRelease(_command_queue);
        _command_queue = nullptr;
      }
      if (_gemm_pso_fp32) {
        CFRelease(_gemm_pso_fp32);
        _gemm_pso_fp32 = nullptr;
      }
      if (_gemm_pso_fp16) {
        CFRelease(_gemm_pso_fp16);
        _gemm_pso_fp16 = nullptr;
      }
      if (_relu_pso) {
        CFRelease(_relu_pso);
        _relu_pso = nullptr;
      }
      if (_gelu_pso) {
        CFRelease(_gelu_pso);
        _gelu_pso = nullptr;
      }
      if (_add_pso) {
        CFRelease(_add_pso);
        _add_pso = nullptr;
      }
      if (_multiply_pso) {
        CFRelease(_multiply_pso);
        _multiply_pso = nullptr;
      }
      if (_layer_norm_pso) {
        CFRelease(_layer_norm_pso);
        _layer_norm_pso = nullptr;
      }
      if (_softmax_pso) {
        CFRelease(_softmax_pso);
        _softmax_pso = nullptr;
      }
      if (_attention_pso) {
        CFRelease(_attention_pso);
        _attention_pso = nullptr;
      }
      if (_flash_attn_pso) {
        CFRelease(_flash_attn_pso);
        _flash_attn_pso = nullptr;
      }
      if (_mps_gemm) {
        CFRelease(_mps_gemm);
        _mps_gemm = nullptr;
      }
      if (_mps_attention) {
        CFRelease(_mps_attention);
        _mps_attention = nullptr;
      }
    }

    std::string MetalDevice::name() const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLDevice> device = queue.device;
        return [device.name UTF8String];
      }
    }

    void MetalDevice::synchronize() {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
      }
    }

    void MetalDevice::synchronize_stream(void* stream) {
      if (stream) {
        @autoreleasepool {
          id<MTLCommandBuffer> commandBuffer = (__bridge id<MTLCommandBuffer>)stream;
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    void* MetalDevice::allocate(std::size_t size) const {
      if (size == 0) {
        return nullptr;
      }

      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLDevice> device = queue.device;
        
        // Use shared storage mode for better compatibility
        MTLResourceOptions options = MTLResourceStorageModeShared;
        
        // Create buffer
        id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
        
        if (!buffer) {
          std::string errorMsg = "Failed to allocate Metal buffer of size " + std::to_string(size);
          std::cerr << errorMsg << std::endl;
          throw std::runtime_error(errorMsg);
        }
        
        return (__bridge_retained void*)buffer;
      }
    }

    void MetalDevice::free(void* data) const {
      if (data) {
        CFRelease(data);
      }
    }

    void* MetalDevice::allocate_stream() const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        return commandBuffer ? (__bridge_retained void*)commandBuffer : nullptr;
      }
    }

    void MetalDevice::free_stream(void* stream) const {
      if (stream) {
        CFRelease(stream);
      }
    }

    void* MetalDevice::create_pipeline_state(const char* kernel_name) const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLDevice> device = queue.device;
        
        NSError* error = nil;
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (!library) {
          throw std::runtime_error("Failed to load Metal library");
        }
        
        id<MTLFunction> kernel = [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
        if (!kernel) {
          throw std::runtime_error(std::string("Failed to find kernel: ") + kernel_name);
        }
        
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:kernel error:&error];
        if (!pso || error) {
          throw std::runtime_error(std::string("Failed to create compute pipeline state for kernel: ") + kernel_name);
        }
        
        return (__bridge_retained void*)pso;
      }
    }

    bool MetalDevice::has_neural_engine() const {
      @autoreleasepool {
        if (@available(macOS 12.0, *)) {
          MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
          config.computeUnits = MLComputeUnitsNeuralEngine;
          return [MLModel deviceHasCapabilityForConfiguration:config];
        }
        return false;
      }
    }

    void MetalDevice::set_prefer_neural_engine(bool prefer) {
      _prefer_neural_engine = prefer;
    }

    bool MetalDevice::prefers_neural_engine() const {
      return _prefer_neural_engine;
    }

    bool MetalDevice::should_use_neural_engine(const char* operation, dim_t size) const {
      // Only use Neural Engine if:
      // 1. It's available
      // 2. We prefer to use it
      // 3. The operation size is large enough to benefit from it
      // 4. We have a CoreML model for this operation
      if (!has_neural_engine() || !_prefer_neural_engine) {
        return false;
      }
      
      // Size thresholds for different operations
      const dim_t GEMM_THRESHOLD = 512;      // Minimum matrix size for GEMM
      const dim_t ATTENTION_THRESHOLD = 256;  // Minimum sequence length for attention
      
      if (strcmp(operation, "gemm") == 0) {
        return size >= GEMM_THRESHOLD && _gemm_mlmodel != nullptr;
      } else if (strcmp(operation, "attention") == 0) {
        return size >= ATTENTION_THRESHOLD && _attention_mlmodel != nullptr;
      }
      
      return false;
    }

    template<typename T>
    void MetalDevice::gemm(T* c,
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
                          void* stream) const {
      @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        
        // Create MPS matrices
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)a
                                                    descriptor:[[MPSMatrixDescriptor alloc] initWithRows:m
                                                                                               columns:k
                                                                                              rowBytes:lda * sizeof(T)
                                                                                              dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16]];
        
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)b
                                                    descriptor:[[MPSMatrixDescriptor alloc] initWithRows:k
                                                                                               columns:n
                                                                                              rowBytes:ldb * sizeof(T)
                                                                                              dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16]];
        
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)c
                                                    descriptor:[[MPSMatrixDescriptor alloc] initWithRows:m
                                                                                               columns:n
                                                                                              rowBytes:ldc * sizeof(T)
                                                                                              dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16]];
        
        // Get MPS GEMM kernel and configure it
        MPSMatrixMultiplication* gemm = (__bridge MPSMatrixMultiplication*)_mps_gemm;
        gemm.alpha = alpha;
        gemm.beta = beta;
        gemm.transpose = transpose_a ? MPSMatrixTransposeYes : MPSMatrixTransposeNo;
        gemm.rightTranspose = transpose_b ? MPSMatrixTransposeYes : MPSMatrixTransposeNo;
        
        // Encode GEMM operation
        [gemm encodeToCommandBuffer:commandBuffer
                        leftMatrix:matrixA
                       rightMatrix:matrixB
                      resultMatrix:matrixC];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    void MetalDevice::gemm_fp32(float* c,
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
      gemm<float>(c, a, b, m, n, k, alpha, beta, lda, ldb, ldc, transpose_a, transpose_b, stream);
    }

    void MetalDevice::gemm_fp16(half* c,
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
                               void* stream) const {
      gemm<half>(c, a, b, m, n, k, alpha, beta, lda, ldb, ldc, transpose_a, transpose_b, stream);
    }

    void MetalDevice::relu(float* output,
                           const float* input,
                           dim_t size,
                           void* stream) const {
      @autoreleasepool {
        if (!_relu_pso) {
          _relu_pso = create_pipeline_state("relu_kernel");
        }
        
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_relu_pso];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
        
        struct {
          uint32_t size;
          float alpha;
          float beta;
        } params = {static_cast<uint32_t>(size), 1.0f, 0.0f};
        
        [computeEncoder setBytes:&params length:sizeof(params) atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    template<typename T>
    void MetalDevice::gelu(T* output,
                          const T* input,
                          dim_t size,
                          void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholder
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          
          // Constants for GELU approximation
          MPSGraphTensor* const_0_5 = [graph constantWithScalar:0.5
                                                     dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          MPSGraphTensor* const_sqrt_2_pi = [graph constantWithScalar:0.7978845608028654
                                                          dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          MPSGraphTensor* const_0_044715 = [graph constantWithScalar:0.044715
                                                          dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // GELU computation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
          MPSGraphTensor* x_cubed = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                         secondaryTensor:[graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                                                              secondaryTensor:inputPlaceholder
                                                                                                       name:nil]
                                                                  name:@"x_cubed"];
          
          MPSGraphTensor* inner_term = [graph multiplicationWithPrimaryTensor:const_0_044715
                                                            secondaryTensor:x_cubed
                                                                     name:@"inner_term"];
          
          MPSGraphTensor* sum_term = [graph additionWithPrimaryTensor:inputPlaceholder
                                                     secondaryTensor:inner_term
                                                              name:@"sum_term"];
          
          MPSGraphTensor* sqrt_term = [graph multiplicationWithPrimaryTensor:const_sqrt_2_pi
                                                           secondaryTensor:sum_term
                                                                    name:@"sqrt_term"];
          
          MPSGraphTensor* tanh_term = [graph tanhWithTensor:sqrt_term
                                                     name:@"tanh_term"];
          
          MPSGraphTensor* add_one = [graph additionWithPrimaryTensor:tanh_term
                                                    secondaryTensor:[graph constantWithScalar:1.0
                                                                                  dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16]
                                                             name:@"add_one"];
          
          MPSGraphTensor* mul_half = [graph multiplicationWithPrimaryTensor:add_one
                                                          secondaryTensor:const_0_5
                                                                   name:@"mul_half"];
          
          MPSGraphTensor* result = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                        secondaryTensor:mul_half
                                                                 name:@"result"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fall back to Metal kernel implementation
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          if (!_gelu_pso) {
            _gelu_pso = create_pipeline_state("gelu_kernel");
          }
          
          id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
          [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_gelu_pso];
          
          // Set buffers
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
          
          // Set parameters
          ElementwiseParams params = {
            static_cast<uint32_t>(size),
            1.0f,
            0.0f
          };
          [computeEncoder setBytes:&params length:sizeof(params) atIndex:2];
          
          // Calculate grid size
          MTLSize gridSize = MTLSizeMake(size, 1, 1);
          MTLSize threadgroupSize = MTLSizeMake(std::min(256u, static_cast<uint32_t>(size)), 1, 1);
          
          // Dispatch compute kernel
          [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
          [computeEncoder endEncoding];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        }
      }
    }

    template<typename T>
    void MetalDevice::relu(T* output,
                          const T* input,
                          dim_t size,
                          void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholder
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          
          // ReLU computation
          MPSGraphTensor* result = [graph reLUWithTensor:inputPlaceholder
                                                  name:@"relu"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fall back to Metal kernel implementation
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          if (!_relu_pso) {
            _relu_pso = create_pipeline_state("relu_kernel");
          }
          
          id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
          [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_relu_pso];
          
          // Set buffers
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
          
          // Set parameters
          ElementwiseParams params = {
            static_cast<uint32_t>(size),
            1.0f,
            0.0f
          };
          [computeEncoder setBytes:&params length:sizeof(params) atIndex:2];
          
          // Calculate grid size
          MTLSize gridSize = MTLSizeMake(size, 1, 1);
          MTLSize threadgroupSize = MTLSizeMake(std::min(256u, static_cast<uint32_t>(size)), 1, 1);
          
          // Dispatch compute kernel
          [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
          [computeEncoder endEncoding];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        }
      }
    }

    void MetalDevice::add(float* output,
                          const float* input1,
                          const float* input2,
                          dim_t size,
                          float alpha,
                          float beta,
                          void* stream) const {
      @autoreleasepool {
        if (!_add_pso) {
          _add_pso = create_pipeline_state("add_kernel");
        }
        
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_add_pso];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input1 offset:0 atIndex:1];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input2 offset:0 atIndex:2];
        
        struct {
          uint32_t size;
          float alpha;
          float beta;
        } params = {
          static_cast<uint32_t>(size),
          alpha,
          beta
        };
        
        [computeEncoder setBytes:&params length:sizeof(params) atIndex:3];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min(256u, static_cast<uint32_t>(size)), 1, 1);
        
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    void MetalDevice::multiply(float* output,
                              const float* input1,
                              const float* input2,
                              dim_t size,
                              float alpha,
                              float beta,
                              void* stream) const {
      @autoreleasepool {
        if (!_multiply_pso) {
          _multiply_pso = create_pipeline_state("multiply_kernel");
        }
        
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_multiply_pso];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input1 offset:0 atIndex:1];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input2 offset:0 atIndex:2];
        
        struct {
          uint32_t size;
          float alpha;
          float beta;
        } params = {
          static_cast<uint32_t>(size),
          alpha,
          beta
        };
        
        [computeEncoder setBytes:&params length:sizeof(params) atIndex:3];
        
        MTLSize gridSize = MTLSizeMake(size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min(256u, static_cast<uint32_t>(size)), 1, 1);
        
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    template<typename T>
    void MetalDevice::layer_norm(T* output,
                               const T* input,
                               const T* gamma,
                               const T* beta,
                               dim_t batch_size,
                               dim_t hidden_size,
                               T epsilon,
                               void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(batch_size), @(hidden_size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* gammaTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)gamma
                                                                                   shape:@[@(hidden_size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* betaTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)beta
                                                                                  shape:@[@(hidden_size)]
                                                                               dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(batch_size), @(hidden_size)]
                                                                  name:@"input"];
          MPSGraphTensor* gammaPlaceholder = [graph placeholderWithShape:@[@(hidden_size)]
                                                                  name:@"gamma"];
          MPSGraphTensor* betaPlaceholder = [graph placeholderWithShape:@[@(hidden_size)]
                                                                 name:@"beta"];
          
          // Compute mean along last dimension
          MPSGraphTensor* mean = [graph meanOfTensor:inputPlaceholder
                                              axes:@[@1]
                                              name:@"mean"];
          mean = [graph expandDimsOfTensor:mean
                                    axis:1
                                    name:@"expanded_mean"];
          
          // Compute variance
          MPSGraphTensor* diff = [graph subtractionWithPrimaryTensor:inputPlaceholder
                                                    secondaryTensor:mean
                                                             name:@"diff"];
          MPSGraphTensor* sqr = [graph squareWithTensor:diff
                                                 name:@"square"];
          MPSGraphTensor* var = [graph meanOfTensor:sqr
                                             axes:@[@1]
                                             name:@"variance"];
          var = [graph expandDimsOfTensor:var
                                   axis:1
                                   name:@"expanded_var"];
          
          // Add epsilon and take sqrt
          MPSGraphTensor* epsilonTensor = [graph constantWithScalar:epsilon
                                                         dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          MPSGraphTensor* varPlusEps = [graph additionWithPrimaryTensor:var
                                                       secondaryTensor:epsilonTensor
                                                                name:@"var_plus_eps"];
          MPSGraphTensor* stddev = [graph squareRootWithTensor:varPlusEps
                                                        name:@"stddev"];
          
          // Normalize
          MPSGraphTensor* normalized = [graph divisionWithPrimaryTensor:diff
                                                      secondaryTensor:stddev
                                                               name:@"normalized"];
          
          // Scale and shift
          MPSGraphTensor* scaled = [graph multiplicationWithPrimaryTensor:normalized
                                                        secondaryTensor:gammaPlaceholder
                                                                 name:@"scaled"];
          MPSGraphTensor* shifted = [graph additionWithPrimaryTensor:scaled
                                                    secondaryTensor:betaPlaceholder
                                                             name:@"shifted"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor,
            gammaPlaceholder: gammaTensor,
            betaPlaceholder: betaTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[shifted]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fall back to Metal kernel implementation
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          if (!_layer_norm_pso) {
            _layer_norm_pso = create_pipeline_state("layer_norm_kernel");
          }
          
          id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
          [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_layer_norm_pso];
          
          // Set buffers
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)gamma offset:0 atIndex:2];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)beta offset:0 atIndex:3];
          
          // Set hidden size parameter
          uint32_t hidden_size_uint = static_cast<uint32_t>(hidden_size);
          [computeEncoder setBytes:&hidden_size_uint length:sizeof(hidden_size_uint) atIndex:4];
          
          // Calculate grid size for batch elements
          MTLSize gridSize = MTLSizeMake(batch_size * hidden_size, 1, 1);
          MTLSize threadgroupSize = MTLSizeMake(std::min(256u, static_cast<uint32_t>(hidden_size)), 1, 1);
          
          // Dispatch compute kernel
          [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
          [computeEncoder endEncoding];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        }
      }
    }

    void MetalDevice::softmax(float* output,
                              const float* input,
                              dim_t batch_size,
                              dim_t feature_size,
                              void* stream) const {
      @autoreleasepool {
        if (!_softmax_pso) {
          _softmax_pso = create_pipeline_state("softmax_kernel");
        }
        
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)_softmax_pso];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)input offset:0 atIndex:1];
        
        struct {
          uint32_t size;
          uint32_t batch_size;
          uint32_t feature_size;
        } params = {
          static_cast<uint32_t>(batch_size * feature_size),
          static_cast<uint32_t>(batch_size),
          static_cast<uint32_t>(feature_size)
        };
        
        [computeEncoder setBytes:&params length:sizeof(params) atIndex:2];
        
        MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min(batch_size, dim_t(256)), 1, 1);
        
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    void MetalDevice::attention(float* output,
                               const float* query,
                               const float* key,
                               const float* value,
                               dim_t batch_size,
                               dim_t num_heads,
                               dim_t head_dim,
                               dim_t seq_len,
                               float scale,
                               bool use_flash,
                               void* stream) const {
      @autoreleasepool {
        void* pso = nullptr;
        const char* kernel_name = use_flash ? "flash_attention_kernel" : "attention_kernel";
        
        if (use_flash) {
          if (!_flash_attn_pso) {
            _flash_attn_pso = create_pipeline_state(kernel_name);
          }
          pso = _flash_attn_pso;
        } else {
          if (!_attention_pso) {
            _attention_pso = create_pipeline_state(kernel_name);
          }
          pso = _attention_pso;
        }
        
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
        id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        
        [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)query offset:0 atIndex:1];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)key offset:0 atIndex:2];
        [computeEncoder setBuffer:(__bridge id<MTLBuffer>)value offset:0 atIndex:3];
        
        struct {
          uint32_t batch_size;
          uint32_t num_heads;
          uint32_t head_dim;
          uint32_t seq_len;
          float scale;
        } params = {
          static_cast<uint32_t>(batch_size),
          static_cast<uint32_t>(num_heads),
          static_cast<uint32_t>(head_dim),
          static_cast<uint32_t>(seq_len),
          scale
        };
        
        [computeEncoder setBytes:&params length:sizeof(params) atIndex:4];
        
        // Allocate shared memory for the attention scores
        NSUInteger sharedMemSize = seq_len * sizeof(float);
        [computeEncoder setThreadgroupMemoryLength:sharedMemSize atIndex:0];
        
        MTLSize gridSize = MTLSizeMake(batch_size, num_heads, seq_len);
        MTLSize threadgroupSize = MTLSizeMake(1, 1, std::min(seq_len, dim_t(32)));
        
        [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [computeEncoder endEncoding];
        
        if (!stream) {
          [commandBuffer commit];
          [commandBuffer waitUntilCompleted];
        }
      }
    }

    template<typename T>
    void MetalDevice::attention(T* output,
                              const T* query,
                              const T* key,
                              const T* value,
                              dim_t batch_size,
                              dim_t num_heads,
                              dim_t head_dim,
                              dim_t seq_len,
                              T scale,
                              void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph and operation
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* queryTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)query
                                                                                   shape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* keyTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)key
                                                                                 shape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                              dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* valueTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)value
                                                                                   shape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders for the operation
          MPSGraphTensor* queryPlaceholder = [graph placeholderWithShape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                  name:@"query"];
          MPSGraphTensor* keyPlaceholder = [graph placeholderWithShape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                name:@"key"];
          MPSGraphTensor* valuePlaceholder = [graph placeholderWithShape:@[@(batch_size), @(num_heads), @(seq_len), @(head_dim)]
                                                                  name:@"value"];
          
          // Create attention operation
          MPSGraphTensor* scaleTensor = [graph constantWithScalar:scale
                                                       dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Compute scaled dot-product attention
          MPSGraphTensor* scores = [graph matrixMultiplicationWithPrimaryTensor:queryPlaceholder
                                                             secondaryTensor:keyPlaceholder
                                                                      name:@"scores"];
          scores = [graph multiplicationWithPrimaryTensor:scores
                                        secondaryTensor:scaleTensor
                                                 name:@"scaled_scores"];
          
          // Apply softmax
          MPSGraphTensor* attentionWeights = [graph softMaxWithTensor:scores
                                                               axis:-1
                                                               name:@"attention_weights"];
          
          // Compute attention output
          MPSGraphTensor* outputTensor = [graph matrixMultiplicationWithPrimaryTensor:attentionWeights
                                                             secondaryTensor:valuePlaceholder
                                                                      name:@"attention_output"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            queryPlaceholder: queryTensor,
            keyPlaceholder: keyTensor,
            valuePlaceholder: valueTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[outputTensor]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fall back to optimized Metal implementation for older macOS versions
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Choose between standard attention and flash attention based on sequence length
          bool use_flash = seq_len > 32;  // Threshold can be tuned
          void* pso = use_flash ? _flash_attn_pso : _attention_pso;
          const char* kernel_name = use_flash ? "flash_attention_kernel" : "attention_kernel";
          
          if (!pso) {
            pso = create_pipeline_state(kernel_name);
            if (use_flash) {
              _flash_attn_pso = pso;
            } else {
              _attention_pso = pso;
            }
          }
          
          id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
          [computeEncoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
          
          // Set buffers
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)output offset:0 atIndex:0];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)query offset:0 atIndex:1];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)key offset:0 atIndex:2];
          [computeEncoder setBuffer:(__bridge id<MTLBuffer>)value offset:0 atIndex:3];
          
          // Set parameters
          AttentionParams params = {
            static_cast<uint32_t>(batch_size),
            static_cast<uint32_t>(num_heads),
            static_cast<uint32_t>(head_dim),
            static_cast<uint32_t>(seq_len),
            static_cast<float>(scale)
          };
          [computeEncoder setBytes:&params length:sizeof(params) atIndex:4];
          
          // Calculate grid and threadgroup sizes
          MTLSize gridSize = MTLSizeMake(batch_size, num_heads, seq_len);
          MTLSize threadgroupSize = MTLSizeMake(1, 1, std::min(32u, static_cast<uint32_t>(seq_len)));
          
          // Dispatch compute kernel
          [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
          [computeEncoder endEncoding];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        }
      }
    }

    template<typename T>
    void MetalDevice::gelu_add(T* output,
                              const T* input,
                              const T* residual,
                              dim_t size,
                              void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* residualTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)residual
                                                                                      shape:@[@(size)]
                                                                                   dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          MPSGraphTensor* residualPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                     name:@"residual"];
          
          // Constants for GELU approximation
          MPSGraphTensor* const_0_5 = [graph constantWithScalar:0.5
                                                     dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          MPSGraphTensor* const_sqrt_2_pi = [graph constantWithScalar:0.7978845608028654
                                                          dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          MPSGraphTensor* const_0_044715 = [graph constantWithScalar:0.044715
                                                          dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // GELU computation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
          MPSGraphTensor* x_cubed = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                         secondaryTensor:[graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                                                              secondaryTensor:inputPlaceholder
                                                                                                       name:nil]
                                                                  name:@"x_cubed"];
          
          MPSGraphTensor* inner_term = [graph multiplicationWithPrimaryTensor:const_0_044715
                                                            secondaryTensor:x_cubed
                                                                     name:@"inner_term"];
          
          MPSGraphTensor* sum_term = [graph additionWithPrimaryTensor:inputPlaceholder
                                                     secondaryTensor:inner_term
                                                              name:@"sum_term"];
          
          MPSGraphTensor* sqrt_term = [graph multiplicationWithPrimaryTensor:const_sqrt_2_pi
                                                           secondaryTensor:sum_term
                                                                    name:@"sqrt_term"];
          
          MPSGraphTensor* tanh_term = [graph tanhWithTensor:sqrt_term
                                                     name:@"tanh_term"];
          
          MPSGraphTensor* add_one = [graph additionWithPrimaryTensor:tanh_term
                                                    secondaryTensor:[graph constantWithScalar:1.0
                                                                                  dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16]
                                                             name:@"add_one"];
          
          MPSGraphTensor* mul_half = [graph multiplicationWithPrimaryTensor:add_one
                                                          secondaryTensor:const_0_5
                                                                   name:@"mul_half"];
          
          MPSGraphTensor* gelu_result = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                            secondaryTensor:mul_half
                                                                     name:@"gelu_result"];
          
          // Add residual
          MPSGraphTensor* result = [graph additionWithPrimaryTensor:gelu_result
                                                   secondaryTensor:residualPlaceholder
                                                            name:@"result"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor,
            residualPlaceholder: residualTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fallback: compute GELU and add separately using existing kernels
          gelu(output, input, size, stream);
          add(output, output, residual, size, stream);
        }
      }
    }

    template<typename T>
    void MetalDevice::relu_add(T* output,
                              const T* input,
                              const T* residual,
                              dim_t size,
                              void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* residualTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)residual
                                                                                      shape:@[@(size)]
                                                                                   dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          MPSGraphTensor* residualPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                     name:@"residual"];
          
          // ReLU computation
          MPSGraphTensor* relu_result = [graph reLUWithTensor:inputPlaceholder
                                                       name:@"relu"];
          
          // Add residual
          MPSGraphTensor* result = [graph additionWithPrimaryTensor:relu_result
                                                   secondaryTensor:residualPlaceholder
                                                            name:@"result"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor,
            residualPlaceholder: residualTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fallback: compute ReLU and add separately using existing kernels
          relu(output, input, size, stream);
          add(output, output, residual, size, stream);
        }
      }
    }

    template<typename T>
    void MetalDevice::dropout(T* output,
                            const T* input,
                            dim_t size,
                            float dropout_prob,
                            bool scale_output,
                            void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          
          // Generate random mask using bernoulli distribution
          MPSGraphTensor* dropoutMask = [graph randomBernoulliWithShape:@[@(size)]
                                                                  rate:1.0 - dropout_prob
                                                                  name:@"dropout_mask"];
          
          // Apply dropout mask
          MPSGraphTensor* maskedInput = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                            secondaryTensor:dropoutMask
                                                                     name:@"masked_input"];
          
          // Scale output if requested
          MPSGraphTensor* result;
          if (scale_output && dropout_prob > 0.0f) {
            float scale = 1.0f / (1.0f - dropout_prob);
            MPSGraphTensor* scaleTensor = [graph constantWithScalar:scale
                                                        dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
            result = [graph multiplicationWithPrimaryTensor:maskedInput
                                         secondaryTensor:scaleTensor
                                                  name:@"scaled_output"];
          } else {
            result = maskedInput;
          }
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fallback implementation for older macOS versions
          std::random_device rd;
          std::mt19937 gen(rd());
          std::bernoulli_distribution dist(1.0 - dropout_prob);
          
          float scale = scale_output && dropout_prob > 0.0f ? 1.0f / (1.0f - dropout_prob) : 1.0f;
          
          for (dim_t i = 0; i < size; ++i) {
            output[i] = dist(gen) ? static_cast<T>(static_cast<float>(input[i]) * scale) : T(0);
          }
        }
      }
    }

    template<typename T>
    void MetalDevice::dropout_add(T* output,
                                const T* input,
                                const T* residual,
                                dim_t size,
                                float dropout_prob,
                                bool scale_output,
                                void* stream) const {
      @autoreleasepool {
        if (@available(macOS 13.0, *)) {
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)_command_queue;
          id<MTLCommandBuffer> commandBuffer = stream ? (__bridge id<MTLCommandBuffer>)stream : [queue commandBuffer];
          
          // Create MPS graph
          MPSGraph* graph = [[MPSGraph alloc] init];
          
          // Create tensor descriptors
          MPSGraphTensorData* inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)input
                                                                                   shape:@[@(size)]
                                                                                dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          MPSGraphTensorData* residualTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)residual
                                                                                      shape:@[@(size)]
                                                                                   dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
          
          // Create placeholders
          MPSGraphTensor* inputPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                  name:@"input"];
          MPSGraphTensor* residualPlaceholder = [graph placeholderWithShape:@[@(size)]
                                                                     name:@"residual"];
          
          // Generate random mask using bernoulli distribution
          MPSGraphTensor* dropoutMask = [graph randomBernoulliWithShape:@[@(size)]
                                                                  rate:1.0 - dropout_prob
                                                                  name:@"dropout_mask"];
          
          // Apply dropout mask
          MPSGraphTensor* maskedInput = [graph multiplicationWithPrimaryTensor:inputPlaceholder
                                                            secondaryTensor:dropoutMask
                                                                     name:@"masked_input"];
          
          // Scale output if requested
          MPSGraphTensor* scaledInput;
          if (scale_output && dropout_prob > 0.0f) {
            float scale = 1.0f / (1.0f - dropout_prob);
            MPSGraphTensor* scaleTensor = [graph constantWithScalar:scale
                                                        dataType:std::is_same<T, float>::value ? MPSDataTypeFloat32 : MPSDataTypeFloat16];
            scaledInput = [graph multiplicationWithPrimaryTensor:maskedInput
                                              secondaryTensor:scaleTensor
                                                       name:@"scaled_input"];
          } else {
            scaledInput = maskedInput;
          }
          
          // Add residual
          MPSGraphTensor* result = [graph additionWithPrimaryTensor:scaledInput
                                                   secondaryTensor:residualPlaceholder
                                                            name:@"result"];
          
          // Create feed dictionary
          NSDictionary* feeds = @{
            inputPlaceholder: inputTensor,
            residualPlaceholder: residualTensor
          };
          
          // Run the graph
          NSArray* results = [graph runWithFeeds:feeds
                                       targetOperations:@[result]
                                     targetTensors:nil
                                    executionDescriptor:nil];
          
          // Copy results back
          MPSGraphTensorData* resultData = results[0];
          [resultData copyToBuffer:(__bridge id<MTLBuffer>)output offset:0];
          
          if (!stream) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
          }
        } else {
          // Fallback implementation for older macOS versions
          std::random_device rd;
          std::mt19937 gen(rd());
          std::bernoulli_distribution dist(1.0 - dropout_prob);
          
          float scale = scale_output && dropout_prob > 0.0f ? 1.0f / (1.0f - dropout_prob) : 1.0f;
          
          for (dim_t i = 0; i < size; ++i) {
            output[i] = (dist(gen) ? static_cast<T>(static_cast<float>(input[i]) * scale) : T(0)) + residual[i];
          }
        }
      }
    }

  }  // namespace metal
}  // namespace ctranslate2
