#include "speaker_profile.h"
#include <algorithm>
#include <simd/simd.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace ctranslate2 {
namespace metal {

SpeakerProfile::SpeakerProfile(const std::string& id)
    : _id(id) {
    // Initialize with default weights
    _weights.separation_weights.resize(1024, 0.0f);  // Example size
    _weights.prediction_weights.resize(1024, 0.0f);  // Example size
}

void SpeakerProfile::update_fingerprint(const VoiceFingerprint& fp) {
    _fingerprint = fp;
    _version++;
}

void SpeakerProfile::update_weights(const ModelWeights& weights) {
    _weights = weights;
    _version++;
}

void SpeakerProfile::update_metrics(float accuracy, float quality) {
    // Exponential moving average for smooth updates
    constexpr float alpha = 0.1f;
    _metrics.prediction_accuracy = (1-alpha) * _metrics.prediction_accuracy + alpha * accuracy;
    _metrics.isolation_quality = (1-alpha) * _metrics.isolation_quality + alpha * quality;
    _metrics.total_processing_time++;
}

void SpeakerProfile::quantize(int8_t target_bits) {
    if (_weights.is_quantized || target_bits >= 32) return;
    
    // Calculate quantization scale
    float scale = float((1 << (target_bits - 1)) - 1);
    
    // Quantize separation weights
    for (auto& w : _weights.separation_weights) {
        w = std::round(w * scale) / scale;
    }
    
    // Quantize prediction weights
    for (auto& w : _weights.prediction_weights) {
        w = std::round(w * scale) / scale;
    }
    
    _weights.quantization_bits = target_bits;
    _weights.is_quantized = true;
    _version++;
}

void SpeakerProfile::dequantize() {
    if (!_weights.is_quantized) return;
    
    // Nothing to do for weights as they're already in float format
    // Just mark as full precision
    _weights.quantization_bits = 32;
    _weights.is_quantized = false;
    _version++;
}

float SpeakerProfile::enhance_audio_quality(const float* input, size_t length, float* output) {
    const size_t vector_size = 4; // Using float4
    const size_t aligned_length = length & ~(vector_size - 1);
    
    // Process vector chunks using SIMD
    for (size_t i = 0; i < aligned_length; i += vector_size) {
        simd_float4 input_vec = simd_load4(input + i);
        simd_float4 output_vec = input_vec;
        
        // Noise gate with SIMD
        simd_float4 abs_vec = simd_abs(output_vec);
        simd_float4 threshold_vec = simd_make_float4(_fingerprint.background_noise_level * 2.0f);
        simd_float4 strength_vec = simd_make_float4(1.0f - background_removal_strength);
        simd_float4 mask = simd_select(simd_make_float4(1.0f), strength_vec, abs_vec < threshold_vec);
        output_vec = output_vec * mask;
        
        // Apply enhancement weights
        size_t weight_idx = i % _weights.separation_weights.size();
        simd_float4 weights_vec = simd_make_float4(
            _weights.separation_weights[weight_idx],
            _weights.separation_weights[(weight_idx + 1) % _weights.separation_weights.size()],
            _weights.separation_weights[(weight_idx + 2) % _weights.separation_weights.size()],
            _weights.separation_weights[(weight_idx + 3) % _weights.separation_weights.size()]
        );
        output_vec = output_vec * weights_vec;
        
        simd_store4(output_vec, output + i);
    }
    
    // Handle remaining samples
    for (size_t i = aligned_length; i < length; i++) {
        output[i] = input[i];
        if (std::abs(output[i]) < _fingerprint.background_noise_level * 2.0f) {
            output[i] *= (1.0f - background_removal_strength);
        }
        size_t weight_idx = i % _weights.separation_weights.size();
        output[i] *= _weights.separation_weights[weight_idx];
    }
    
    return calculate_enhancement_quality(input, output, length);
}

float SpeakerProfile::reconstruct_audio(const float* degraded_input, size_t length, float* enhanced_output) {
    const size_t vector_size = 4;
    const size_t aligned_length = length & ~(vector_size - 1);
    
    // Process vector chunks using SIMD
    for (size_t i = 0; i < aligned_length; i += vector_size) {
        simd_float4 input_vec = simd_load4(degraded_input + i);
        
        // Apply prediction weights
        size_t weight_idx = i % _weights.prediction_weights.size();
        simd_float4 pred_weights = simd_make_float4(
            _weights.prediction_weights[weight_idx],
            _weights.prediction_weights[(weight_idx + 1) % _weights.prediction_weights.size()],
            _weights.prediction_weights[(weight_idx + 2) % _weights.prediction_weights.size()],
            _weights.prediction_weights[(weight_idx + 3) % _weights.prediction_weights.size()]
        );
        
        simd_float4 output_vec = input_vec * pred_weights;
        
        // Apply frequency correction if available
        if (_fingerprint.frequency_response.size() > 0) {
            simd_float4 freq_weights = simd_make_float4(
                _fingerprint.frequency_response[i % _fingerprint.frequency_response.size()],
                _fingerprint.frequency_response[(i + 1) % _fingerprint.frequency_response.size()],
                _fingerprint.frequency_response[(i + 2) % _fingerprint.frequency_response.size()],
                _fingerprint.frequency_response[(i + 3) % _fingerprint.frequency_response.size()]
            );
            output_vec = output_vec * freq_weights;
        }
        
        simd_store4(output_vec, enhanced_output + i);
    }
    
    // Handle remaining samples
    for (size_t i = aligned_length; i < length; i++) {
        enhanced_output[i] = degraded_input[i];
        size_t weight_idx = i % _weights.prediction_weights.size();
        enhanced_output[i] *= _weights.prediction_weights[weight_idx];
        
        if (_fingerprint.frequency_response.size() > 0) {
            size_t freq_idx = i % _fingerprint.frequency_response.size();
            enhanced_output[i] *= _fingerprint.frequency_response[freq_idx];
        }
    }
    
    return calculate_enhancement_quality(degraded_input, enhanced_output, length);
}

float SpeakerProfile::calculate_enhancement_quality(const float* original, const float* enhanced, size_t length) {
    const size_t vector_size = 4;
    const size_t aligned_length = length & ~(vector_size - 1);
    
    simd_float4 sum_squared_error_vec = simd_make_float4(0.0f);
    simd_float4 sum_squared_original_vec = simd_make_float4(0.0f);
    
    // Process vector chunks
    for (size_t i = 0; i < aligned_length; i += vector_size) {
        simd_float4 original_vec = simd_load4(original + i);
        simd_float4 enhanced_vec = simd_load4(enhanced + i);
        simd_float4 error_vec = enhanced_vec - original_vec;
        
        sum_squared_error_vec += error_vec * error_vec;
        sum_squared_original_vec += original_vec * original_vec;
    }
    
    // Reduce vectors
    float sum_squared_error = simd_reduce_add(sum_squared_error_vec);
    float sum_squared_original = simd_reduce_add(sum_squared_original_vec);
    
    // Handle remaining samples
    for (size_t i = aligned_length; i < length; i++) {
        float error = enhanced[i] - original[i];
        sum_squared_error += error * error;
        sum_squared_original += original[i] * original[i];
    }
    
    return 10.0f * log10f(sum_squared_original / (sum_squared_error + 1e-10f));
}

CompressedAudioFrame SpeakerProfile::compress_audio(const float* input, size_t length) {
    CompressedAudioFrame frame;
    frame.original_length = length;
    
    // First enhance the audio using our profile
    std::vector<float> enhanced(length);
    enhance_audio_quality(input, length, enhanced.data());
    
    // Check for speech content
    frame.contains_speech = is_speech_frame(enhanced.data(), length);
    frame.speech_confidence = calculate_speech_probability(enhanced.data(), length);
    
    // If it's mostly silence, use aggressive compression
    if (!frame.contains_speech) {
        compression_config.bits_per_sample = 2;  // Very low bit depth for silence
    } else {
        compression_config.bits_per_sample = compression_config.bits_per_sample;  // Use normal setting
    }
    
    // Encode the frame
    frame.compressed_data = encode_frame(enhanced.data(), length);
    frame.compression_ratio = float(length * sizeof(float)) / frame.compressed_data.size();
    
    return frame;
}

std::vector<float> SpeakerProfile::decompress_audio(const CompressedAudioFrame& frame) {
    // Decode the compressed data
    std::vector<float> decompressed = decode_frame(frame.compressed_data, frame.original_length);
    
    // If it contained speech, apply profile-based enhancement
    if (frame.contains_speech) {
        std::vector<float> enhanced(frame.original_length);
        enhance_audio_quality(decompressed.data(), frame.original_length, enhanced.data());
        return enhanced;
    }
    
    return decompressed;
}

bool SpeakerProfile::is_speech_frame(const float* frame, size_t length) {
    float energy = 0.0f;
    const size_t vector_size = 4;
    const size_t aligned_length = length & ~(vector_size - 1);
    
    // Process vector chunks
    simd_float4 energy_vec = simd_make_float4(0.0f);
    for (size_t i = 0; i < aligned_length; i += vector_size) {
        simd_float4 frame_vec = simd_load4(frame + i);
        energy_vec += frame_vec * frame_vec;
    }
    energy = simd_reduce_add(energy_vec);
    
    // Handle remaining samples
    for (size_t i = aligned_length; i < length; i++) {
        energy += frame[i] * frame[i];
    }
    
    energy /= length;
    return energy > compression_config.speech_threshold;
}

float SpeakerProfile::calculate_speech_probability(const float* frame, size_t length) {
    float energy = 0.0f;
    const size_t vector_size = 4;
    const size_t aligned_length = length & ~(vector_size - 1);
    
    // Process vector chunks
    simd_float4 energy_vec = simd_make_float4(0.0f);
    for (size_t i = 0; i < aligned_length; i += vector_size) {
        simd_float4 frame_vec = simd_load4(frame + i);
        energy_vec += frame_vec * frame_vec;
    }
    energy = simd_reduce_add(energy_vec);
    
    // Handle remaining samples
    for (size_t i = aligned_length; i < length; i++) {
        energy += frame[i] * frame[i];
    }
    
    energy /= length;
    
    // Calculate probability using sigmoid function
    float x = (energy - compression_config.speech_threshold) / compression_config.speech_threshold;
    return 1.0f / (1.0f + std::exp(-10.0f * x));  // Steeper sigmoid for clearer decision
}

std::vector<uint8_t> SpeakerProfile::encode_frame(const float* input, size_t length) {
    std::vector<uint8_t> compressed;
    compressed.reserve(length * compression_config.bits_per_sample / 8);
    
    // Implement your compression logic here
    // This is a placeholder implementation
    for (size_t i = 0; i < length; i++) {
        float normalized = std::max(-1.0f, std::min(1.0f, input[i]));
        uint8_t quantized = uint8_t((normalized + 1.0f) * 127.5f);
        compressed.push_back(quantized);
    }
    
    return compressed;
}

std::vector<float> SpeakerProfile::decode_frame(const std::vector<uint8_t>& compressed, size_t original_length) {
    std::vector<float> decompressed(original_length);
    
    // Implement your decompression logic here
    // This is a placeholder implementation
    for (size_t i = 0; i < original_length && i < compressed.size(); i++) {
        decompressed[i] = (float(compressed[i]) / 127.5f) - 1.0f;
    }
    
    return decompressed;
}

void SpeakerProfile::start_background_reprocessing() {
    if (!_background_processing.load()) {
        _background_processing.store(true);
        std::thread reprocessing_thread(&SpeakerProfile::background_reprocessing_thread, this);
        reprocessing_thread.detach();
    }
}

void SpeakerProfile::stop_background_reprocessing() {
    _background_processing.store(false);
}

float SpeakerProfile::get_reprocessing_progress() const {
    return _reprocessing_progress.load();
}

void SpeakerProfile::background_reprocessing_thread() {
    size_t total_files = _history.processed_files.size();
    size_t processed = 0;
    
    for (const auto& file : _history.processed_files) {
        if (!_background_processing.load()) break;
        
        if (needs_reprocessing(file)) {
            reprocess_with_current_model(file);
        }
        
        processed++;
        _reprocessing_progress.store(float(processed) / total_files);
    }
    
    _background_processing.store(false);
    _reprocessing_progress.store(1.0f);
}

} // namespace metal
} // namespace ctranslate2
