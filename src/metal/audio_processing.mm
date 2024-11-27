#include "audio_processing.h"
#include <cstring>

namespace ctranslate2 {
namespace metal {

RingBuffer::RingBuffer(size_t capacity) 
  : _capacity(capacity)
  , _read_pos(0)
  , _write_pos(0)
  , _count(0) {
  if (__builtin_available(macOS 10.15, *)) {
    _data = static_cast<float*>(aligned_alloc(16, capacity * sizeof(float)));
  } else {
    _data = static_cast<float*>(malloc(capacity * sizeof(float)));
  }
}

RingBuffer::~RingBuffer() {
  free(_data);
}

void RingBuffer::push(const float* data, size_t count) {
  size_t remaining = _capacity - _count;
  if (count > remaining) {
    count = remaining;
  }
  size_t write_pos = _write_pos.load(std::memory_order_relaxed);
  size_t first_write = std::min(count, _capacity - write_pos);
  
  memcpy(_data + write_pos, data, first_write * sizeof(float));
  if (first_write < count) {
    memcpy(_data, data + first_write, (count - first_write) * sizeof(float));
  }
  
  _write_pos.store((write_pos + count) % _capacity, std::memory_order_release);
  _count += count;
}

void RingBuffer::pop(float* data, size_t count) {
  size_t available = _count;
  if (count > available) {
    count = available;
  }
  size_t read_pos = _read_pos.load(std::memory_order_relaxed);
  size_t first_read = std::min(count, _capacity - read_pos);
  
  memcpy(data, _data + read_pos, first_read * sizeof(float));
  if (first_read < count) {
    memcpy(data + first_read, _data, (count - first_read) * sizeof(float));
  }
  
  _read_pos.store((read_pos + count) % _capacity, std::memory_order_release);
  _count -= count;
}

AudioProcessor::AudioProcessor(id<MTLDevice> device, id<MTLCommandQueue> commandQueue)
  : _device(device)
  , _commandQueue(commandQueue)
  , _use_neural_engine(false)
  , _power_mode(PowerMode::Balanced)
  , _input_ring_buffer(kRingBufferSize)
  , _output_ring_buffer(kRingBufferSize)
  , _sample_rate(16000.0f)  // Default to 16kHz
  , _lowpass_cutoff(8000.0f / _sample_rate)  // Default 8kHz lowpass
  , _highpass_cutoff(20.0f / _sample_rate) {  // Default 20Hz highpass
  
  NSError* error = nil;
  
  // Create compute pipeline
  id<MTLLibrary> library = [_device newDefaultLibrary];
  id<MTLFunction> kernelFunction = [library newFunctionWithName:@"process_audio_simd"];
  _pipelineState = [_device newComputePipelineStateWithFunction:kernelFunction error:&error];
  
  if (!_pipelineState) {
    NSLog(@"Failed to create pipeline state: %@", error);
    return;
  }
  
  // Calculate number of filter states needed
  const size_t simd_size = 4;
  const size_t num_vectors = (kProcessingChunkSize + simd_size - 1) / simd_size;
  const size_t num_states = num_vectors * 2;  // 2 filters per vector
  
  // Create and initialize filter states buffer
  _filter_states_buffer = [_device newBufferWithLength:sizeof(FilterState) * num_states
                                             options:MTLResourceStorageModeShared];
  reset_filter_states();
}

AudioProcessor::~AudioProcessor() {
  if (_filter_states_buffer) {
    [_filter_states_buffer release];
  }
}

void AudioProcessor::reset_filter_states() {
  FilterState* states = static_cast<FilterState*>([_filter_states_buffer contents]);
  const size_t simd_size = 4;
  const size_t num_vectors = (kProcessingChunkSize + simd_size - 1) / simd_size;
  const size_t num_states = num_vectors * 2;
  
  for (size_t i = 0; i < num_states; ++i) {
    memset(&states[i], 0, sizeof(FilterState));
  }
}

void AudioProcessor::set_power_mode(PowerMode mode) {
  _power_mode = mode;
  
  // Adjust processing parameters based on power mode
  switch (mode) {
    case PowerMode::LowPower:
      _use_neural_engine = false;
      break;
    case PowerMode::HighPerformance:
      _use_neural_engine = true;
      break;
    case PowerMode::Balanced:
      _use_neural_engine = false;
      break;
  }
}

void AudioProcessor::process_audio(const float* input, size_t input_size, float* output, size_t output_size) {
  _input_ring_buffer.push(input, input_size);
  
  while (process_available()) {
    // Processing happens in process_available()
  }
  
  _output_ring_buffer.pop(output, output_size);
}

bool AudioProcessor::process_available() {
  if (_input_ring_buffer.available_read() < kProcessingChunkSize) {
    return false;
  }
  
  if (_output_ring_buffer.available_write() < kProcessingChunkSize) {
    return false;
  }
  
  float input_chunk[kProcessingChunkSize];
  float output_chunk[kProcessingChunkSize];
  
  _input_ring_buffer.pop(input_chunk, kProcessingChunkSize);
  
  // Calculate SIMD size (float4)
  const size_t simd_size = 4;
  const size_t num_vectors = (kProcessingChunkSize + simd_size - 1) / simd_size;
  
  // Create input buffer with proper alignment for SIMD
  id<MTLBuffer> inputBuffer = [_device newBufferWithLength:num_vectors * sizeof(float) * simd_size
                                                 options:MTLResourceStorageModeShared];
  float* input_ptr = (float*)[inputBuffer contents];
  memcpy(input_ptr, input_chunk, kProcessingChunkSize * sizeof(float));
  
  // Create output buffer
  id<MTLBuffer> outputBuffer = [_device newBufferWithLength:num_vectors * sizeof(float) * simd_size
                                                  options:MTLResourceStorageModeShared];
  
  // Create parameters buffer
  float params[2] = {_lowpass_cutoff, _highpass_cutoff};
  id<MTLBuffer> paramsBuffer = [_device newBufferWithBytes:params
                                                  length:sizeof(params)
                                                 options:MTLResourceStorageModeShared];
  
  // Create command buffer
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  
  // Use compute kernel
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  [encoder setComputePipelineState:_pipelineState];
  [encoder setBuffer:inputBuffer offset:0 atIndex:0];
  [encoder setBuffer:outputBuffer offset:0 atIndex:1];
  [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
  [encoder setBuffer:_filter_states_buffer offset:0 atIndex:3];
  
  // Adjust grid size for SIMD processing
  MTLSize gridSize = MTLSizeMake(num_vectors, 1, 1);
  NSUInteger threadGroupSize = MIN(_pipelineState.maxTotalThreadsPerThreadgroup, num_vectors);
  MTLSize threadGroupSizeObj = MTLSizeMake(threadGroupSize, 1, 1);
  [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSizeObj];
  [encoder endEncoding];
  
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  
  // Copy result back, only copying the valid data
  float* output_ptr = (float*)[outputBuffer contents];
  memcpy(output_chunk, output_ptr, kProcessingChunkSize * sizeof(float));
  
  // Release resources
  [inputBuffer release];
  [outputBuffer release];
  [paramsBuffer release];
  
  _output_ring_buffer.push(output_chunk, kProcessingChunkSize);
  
  return true;
}

} // namespace metal
} // namespace ctranslate2
