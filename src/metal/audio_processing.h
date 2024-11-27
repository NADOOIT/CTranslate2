#pragma once

#include <atomic>
#include <vector>
#include <Metal/Metal.h>

namespace ctranslate2 {
namespace metal {

const size_t kRingBufferSize = 16000;  // 1 second at 16kHz
const size_t kProcessingChunkSize = 256; // 16ms at 16kHz

// Filter state structure matching Metal shader
struct FilterState {
    float x1[4];  // Previous input sample
    float x2[4];  // Second previous input sample
    float y1[4];  // Previous output sample
    float y2[4];  // Second previous output sample
};

enum class PowerMode {
  LowPower,
  HighPerformance,
  Balanced
};

class RingBuffer {
public:
  RingBuffer(size_t capacity);
  ~RingBuffer();

  void push(const float* data, size_t count);
  void pop(float* data, size_t count);
  
  size_t available_read() const { return _count; }
  size_t available_write() const { return _capacity - _count; }

private:
  float* _data;
  size_t _capacity;
  std::atomic<size_t> _read_pos;
  std::atomic<size_t> _write_pos;
  size_t _count;
};

class AudioProcessor {
public:
  AudioProcessor(id<MTLDevice> device, id<MTLCommandQueue> commandQueue);
  ~AudioProcessor();

  void set_power_mode(PowerMode mode);
  void process_audio(const float* input, size_t input_size, float* output, size_t output_size);
  
  // Filter parameter setters (cutoff frequencies in Hz, will be normalized internally)
  void set_lowpass_cutoff(float cutoff_hz) { 
    _lowpass_cutoff = cutoff_hz / _sample_rate;
  }
  void set_highpass_cutoff(float cutoff_hz) { 
    _highpass_cutoff = cutoff_hz / _sample_rate;
  }

private:
  bool process_available();
  void reset_filter_states();

  id<MTLDevice> _device;
  id<MTLCommandQueue> _commandQueue;
  bool _use_neural_engine;
  PowerMode _power_mode;
  RingBuffer _input_ring_buffer;
  RingBuffer _output_ring_buffer;
  id<MTLComputePipelineState> _pipelineState;
  id<MTLBuffer> _filter_states_buffer;
  
  // Filter parameters
  float _sample_rate;
  float _lowpass_cutoff;  // Normalized cutoff frequency (0-1)
  float _highpass_cutoff; // Normalized cutoff frequency (0-1)
};

} // namespace metal
} // namespace ctranslate2
