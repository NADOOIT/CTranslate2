#include "speaker_manager.h"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <simd/simd.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include "nlohmann/json.hpp"
#include "zlib.h"

namespace ctranslate2 {
namespace metal {

namespace {
    // Utility functions for fingerprint comparison
    float cosine_similarity(const MelFingerprint& a, const MelFingerprint& b) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
    
    // Compress model data
    std::vector<uint8_t> compress_data(const std::string& data) {
        std::vector<uint8_t> compressed;
        compressed.resize(data.size() + 128); // Some headroom
        
        z_stream stream;
        stream.zalloc = Z_NULL;
        stream.zfree = Z_NULL;
        stream.opaque = Z_NULL;
        
        deflateInit(&stream, Z_BEST_COMPRESSION);
        
        stream.next_in = (Bytef*)data.data();
        stream.avail_in = data.size();
        stream.next_out = compressed.data();
        stream.avail_out = compressed.size();
        
        deflate(&stream, Z_FINISH);
        deflateEnd(&stream);
        
        compressed.resize(stream.total_out);
        return compressed;
    }
    
    // Decompress model data
    std::string decompress_data(const std::vector<uint8_t>& compressed) {
        std::string decompressed;
        decompressed.resize(compressed.size() * 4); // Estimate
        
        z_stream stream;
        stream.zalloc = Z_NULL;
        stream.zfree = Z_NULL;
        stream.opaque = Z_NULL;
        
        inflateInit(&stream);
        
        stream.next_in = (Bytef*)compressed.data();
        stream.avail_in = compressed.size();
        stream.next_out = (Bytef*)decompressed.data();
        stream.avail_out = decompressed.size();
        
        inflate(&stream, Z_FINISH);
        inflateEnd(&stream);
        
        decompressed.resize(stream.total_out);
        return decompressed;
    }
}

SpeakerManager::SpeakerManager(id<MTLDevice> device, id<MTLCommandQueue> command_queue)
    : _device(device)
    , _command_queue(command_queue) {
    setup_neural_engine_pipeline();
}

SpeakerManager::~SpeakerManager() {
    [_fingerprint_pipeline release];
}

void SpeakerManager::setup_neural_engine_pipeline() {
    // Create a neural network for fingerprint extraction
    MPSNNImageNode* input = [MPSNNImageNode nodeWithHandle:nil];
    
    // Convert audio to mel spectrogram
    MPSNNFilterNode* melNode = [MPSNNFilterNode nodeWithSource:input
                                                     weights:[[MPSNNWeights alloc] init]];
    
    // Extract features
    MPSCNNNeuronReLUNode* featureNode = [MPSCNNNeuronReLUNode nodeWithSource:melNode];
    
    // Create graph
    _fingerprint_pipeline = [MPSNNGraph graphWithDevice:_device
                                                input:input
                                               output:featureNode];
}

VoiceFingerprint SpeakerManager::extract_fingerprint(const float* audio, size_t size) {
    VoiceFingerprint fp;
    
    // Create input buffer
    id<MTLBuffer> inputBuffer = [_device newBufferWithBytes:audio
                                                   length:size * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    
    // Create output buffer for mel spectrogram
    id<MTLBuffer> outputBuffer = [_device newBufferWithLength:kMelBins * kTimeFrames * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    
    // Process through neural engine
    [_fingerprint_pipeline encodeToCommandBuffer:[_command_queue commandBuffer]
                                    sourceImage:inputBuffer
                                 destinationImage:outputBuffer];
    
    // Copy to fingerprint
    float* output_ptr = static_cast<float*>([outputBuffer contents]);
    std::copy(output_ptr, output_ptr + kMelBins * kTimeFrames, fp.mel_fingerprint.begin());
    
    // Calculate additional features
    calculate_pitch_contour(audio, size, fp.pitch_contour);
    fp.speaking_rate = calculate_speaking_rate(audio, size);
    fp.fundamental_freq = calculate_fundamental_frequency(audio, size);
    
    [inputBuffer release];
    [outputBuffer release];
    
    return fp;
}

std::string SpeakerManager::identify_speaker(const float* audio, size_t size) {
    auto fp = extract_fingerprint(audio, size);
    return match_fingerprint(fp);
}

std::string SpeakerManager::match_fingerprint(const VoiceFingerprint& fp) {
    std::lock_guard<std::mutex> lock(_profiles_mutex);
    
    float best_similarity = 0.0f;
    std::string best_match;
    
    for (const auto& [id, profile] : _active_profiles) {
        float similarity = cosine_similarity(fp.mel_fingerprint, 
                                          profile.fingerprint().mel_fingerprint);
        
        if (similarity > best_similarity && similarity > 0.85f) { // Threshold
            best_similarity = similarity;
            best_match = id;
        }
    }
    
    return best_match.empty() ? "unknown_speaker" : best_match;
}

bool SpeakerManager::save_speaker_model(const std::string& speaker_id) {
    std::lock_guard<std::mutex> lock(_profiles_mutex);
    
    auto it = _active_profiles.find(speaker_id);
    if (it == _active_profiles.end()) return false;
    
    const auto& profile = it->second;
    
    // Create JSON representation
    nlohmann::json model_json = {
        {"speaker_id", speaker_id},
        {"fingerprint", {
            {"mel_fingerprint", profile.fingerprint().mel_fingerprint},
            {"pitch_contour", profile.fingerprint().pitch_contour},
            {"speaking_rate", profile.fingerprint().speaking_rate},
            {"fundamental_freq", profile.fingerprint().fundamental_freq}
        }},
        {"weights", {
            {"separation", profile.weights().separation_weights},
            {"prediction", profile.weights().prediction_weights}
        }},
        {"version", profile.version()},
        {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()}
    };
    
    // Compress and save
    try {
        std::string path = get_model_path(speaker_id);
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        auto compressed = compress_data(model_json.dump());
        file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
        
        return true;
    } catch (...) {
        return false;
    }
}

bool SpeakerManager::load_speaker_model(const std::string& speaker_id) {
    try {
        std::string path = get_model_path(speaker_id);
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Read compressed data
        std::vector<uint8_t> compressed_data((std::istreambuf_iterator<char>(file)),
                                           std::istreambuf_iterator<char>());
        
        // Decompress and parse JSON
        std::string json_str = decompress_data(compressed_data);
        auto model_json = nlohmann::json::parse(json_str);
        
        // Create new profile
        SpeakerProfile profile(speaker_id);
        
        // Load fingerprint
        VoiceFingerprint fp;
        fp.mel_fingerprint = model_json["fingerprint"]["mel_fingerprint"].get<MelFingerprint>();
        fp.pitch_contour = model_json["fingerprint"]["pitch_contour"].get<std::vector<float>>();
        fp.speaking_rate = model_json["fingerprint"]["speaking_rate"].get<float>();
        fp.fundamental_freq = model_json["fingerprint"]["fundamental_freq"].get<float>();
        profile.update_fingerprint(fp);
        
        // Load weights
        ModelWeights weights;
        weights.separation_weights = model_json["weights"]["separation"].get<std::vector<float>>();
        weights.prediction_weights = model_json["weights"]["prediction"].get<std::vector<float>>();
        profile.update_weights(weights);
        
        // Store profile
        std::lock_guard<std::mutex> lock(_profiles_mutex);
        _active_profiles[speaker_id] = std::move(profile);
        
        return true;
    } catch (...) {
        return false;
    }
}

void SpeakerManager::create_new_speaker_profile(const std::string& speaker_id) {
    std::lock_guard<std::mutex> lock(_profiles_mutex);
    _active_profiles.emplace(speaker_id, SpeakerProfile(speaker_id));
}

void SpeakerManager::process_audio_with_speaker_detection(const float* audio, size_t size) {
    std::string detected_speaker = identify_speaker(audio, size);
    _current_speaker_id = detected_speaker;
    
    if (detected_speaker != "unknown_speaker") {
        std::lock_guard<std::mutex> lock(_profiles_mutex);
        auto it = _active_profiles.find(detected_speaker);
        if (it != _active_profiles.end()) {
            process_with_model(audio, size, it->second);
        }
    }
}

void SpeakerManager::process_with_model(const float* audio, size_t size, SpeakerProfile& profile) {
    // Create buffers
    id<MTLBuffer> inputBuffer = [_device newBufferWithBytes:audio
                                                   length:size * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> outputBuffer = [_device newBufferWithLength:size * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    
    // Process audio using profile's model
    float* output_ptr = static_cast<float*>([outputBuffer contents]);
    profile.enhance_audio_quality(audio, size, output_ptr);
    
    // Update profile metrics
    float quality = calculate_output_quality(audio, output_ptr, size);
    profile.update_metrics(1.0f, quality);  // Assuming perfect prediction for now
    
    [inputBuffer release];
    [outputBuffer release];
}

std::optional<SpeakerProfile> SpeakerManager::get_current_profile() const {
    std::lock_guard<std::mutex> lock(_profiles_mutex);
    auto it = _active_profiles.find(_current_speaker_id);
    if (it != _active_profiles.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::string SpeakerManager::get_model_path(const std::string& speaker_id) {
    // Implement your path generation logic here
    return "models/speaker_" + speaker_id + ".model";
}

} // namespace metal
} // namespace ctranslate2
