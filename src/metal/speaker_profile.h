#ifndef CTRANSLATE2_METAL_SPEAKER_PROFILE_H_
#define CTRANSLATE2_METAL_SPEAKER_PROFILE_H_

#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <memory>
#include <array>
#include <atomic>

namespace ctranslate2 {
namespace metal {

// Fixed-size mel spectrogram fingerprint
constexpr size_t kMelBins = 80;
constexpr size_t kTimeFrames = 100;
using MelFingerprint = std::array<float, kMelBins * kTimeFrames>;

struct VoiceFingerprint {
    MelFingerprint mel_fingerprint;
    std::vector<float> pitch_contour;
    float speaking_rate;
    std::string primary_language;
    float fundamental_freq;
    
    // Serialization
    template<class Archive>
    void serialize(Archive& ar) {
        ar(mel_fingerprint, pitch_contour, speaking_rate, 
           primary_language, fundamental_freq);
    }
};

struct ModelWeights {
    std::vector<float> separation_weights;
    std::vector<float> prediction_weights;
    int8_t quantization_bits = 32;  // Default to full precision
    bool is_quantized = false;
    
    template<class Archive>
    void serialize(Archive& ar) {
        ar(separation_weights, prediction_weights, quantization_bits, is_quantized);
    }
};

struct CompressedAudioFrame {
    std::vector<uint8_t> compressed_data;
    float compression_ratio;
    uint32_t original_length;
    bool contains_speech;
    float speech_confidence;
};

struct WordDetectionResult {
    float start_time;
    float end_time;
    float confidence;
    std::string word;
    bool is_filtered;  // True if enhanced by profile
};

class SpeakerProfile {
public:
    SpeakerProfile() = default;
    explicit SpeakerProfile(const std::string& id);
    
    // Core profile data
    const std::string& id() const { return _id; }
    const VoiceFingerprint& fingerprint() const { return _fingerprint; }
    const ModelWeights& weights() const { return _weights; }
    
    // Profile management
    void update_fingerprint(const VoiceFingerprint& fp);
    void update_weights(const ModelWeights& weights);
    uint64_t version() const { return _version; }
    
    // Model performance metrics
    float get_prediction_accuracy() const { return _metrics.prediction_accuracy; }
    void update_metrics(float accuracy, float quality);

    // New methods for quantization and enhancement
    void quantize(int8_t target_bits = 8);
    void dequantize();
    
    // Audio enhancement methods
    float enhance_audio_quality(const float* input, size_t length, float* output);
    float reconstruct_audio(const float* degraded_input, size_t length, float* enhanced_output);
    
    // New methods for compression and word detection
    CompressedAudioFrame compress_audio(const float* input, size_t length);
    std::vector<float> decompress_audio(const CompressedAudioFrame& frame);
    
    // Enhanced word detection with profile-based filtering
    std::vector<WordDetectionResult> detect_words(const float* audio, size_t length, bool use_enhancement = true);
    
    enum class ProcessingMode {
        FAST_STARTUP,    // Basic processing, no enhancement
        TRANSITIONING,   // Building profile, partial enhancement
        FULL_ENHANCED   // Full SIMD + profile enhancement
    };

    struct ProcessingState {
        ProcessingMode current_mode = ProcessingMode::FAST_STARTUP;
        size_t processed_frames = 0;
        size_t transition_threshold = 1000;  // Frames before transition
        bool simd_available = false;
        float enhancement_ratio = 0.0f;  // 0.0 to 1.0
    } _processing_state;

    // Progressive processing methods
    void update_processing_mode();
    float* process_audio_progressive(float* audio, size_t length);
    
    // Different processing implementations
    float* process_audio_basic(float* audio, size_t length);
    float* process_audio_simd(float* audio, size_t length);
    float* process_audio_enhanced(float* audio, size_t length);
    
    struct SmartCompressionConfig {
        // Compression modes
        enum class Mode {
            ULTRA_COMPACT,    // Maximum compression, good for archival
            BALANCED,         // Good compression while maintaining quality
            HIGH_QUALITY      // Minimal compression, best quality
        };
        
        Mode mode = Mode::BALANCED;
        bool use_profile_prediction = true;    // Use profile for better compression
        bool store_fingerprints = true;        // Store voice fingerprints for reconstruction
        uint8_t silence_bits = 2;              // Bits used for silence
        uint8_t speech_bits = 8;               // Bits used for speech
        float quality_threshold = 0.95f;       // Quality threshold for compression
    };
    
    struct CompressedAudioFile {
        std::vector<CompressedAudioFrame> frames;
        VoiceFingerprint reference_fingerprint;
        std::vector<float> frequency_response;
        float original_duration;
        float compression_ratio;
        uint32_t sample_rate;
        std::string profile_id;
    };
    
    // Compression methods
    CompressedAudioFile compress_file(const std::string& input_path, 
                                    const SmartCompressionConfig& config = SmartCompressionConfig());
    
    void save_compressed(const std::string& output_path, 
                        const CompressedAudioFile& compressed);
    
    CompressedAudioFile load_compressed(const std::string& input_path);
    
    void decompress_file(const CompressedAudioFile& compressed,
                        const std::string& output_path);
                        
    // Quality assessment
    float estimate_compression_quality(const CompressedAudioFile& compressed);
    
    struct ProcessingHistory {
        std::vector<std::string> processed_files;
        std::map<std::string, float> quality_scores;
        uint64_t profile_version_at_processing;
        bool needs_reprocessing;
    };
    
    // Retrospective processing
    struct ReprocessingResult {
        size_t improved_segments;
        float quality_improvement;
        std::vector<WordDetectionResult> new_words;
        std::vector<WordDetectionResult> corrected_words;
    };
    
    // History tracking
    void track_processing(const std::string& file_path, float quality_score);
    bool needs_reprocessing(const std::string& file_path) const;
    
    // Retrospective enhancement
    ReprocessingResult reprocess_with_current_model(const std::string& file_path);
    ReprocessingResult batch_reprocess_all();
    
    // Background processing
    void start_background_reprocessing();
    void stop_background_reprocessing();
    float get_reprocessing_progress() const;
    
private:
    std::string _id;
    VoiceFingerprint _fingerprint;
    ModelWeights _weights;
    uint64_t _version{0};
    
    struct Metrics {
        float prediction_accuracy{0.0f};
        float isolation_quality{0.0f};
        uint64_t total_processing_time{0};
    } _metrics;
    
    // Background removal strength (0.0 to 1.0)
    float background_removal_strength = 0.95f;
    
    // Profile-based audio compression settings
    struct CompressionConfig {
        float silence_threshold = 0.01f;
        float speech_threshold = 0.1f;
        uint8_t bits_per_sample = 4;  // Adaptive bit depth
        bool use_predictive_coding = true;
    } compression_config;
    
    // Internal methods for compression
    std::vector<uint8_t> encode_frame(const float* input, size_t length);
    std::vector<float> decode_frame(const std::vector<uint8_t>& compressed, size_t original_length);
    
    // Speech detection helpers
    bool is_speech_frame(const float* frame, size_t length);
    float calculate_speech_probability(const float* frame, size_t length);
    
    ProcessingHistory _history;
    std::atomic<bool> _background_processing{false};
    std::atomic<float> _reprocessing_progress{0.0f};
    
    void background_reprocessing_thread();
};

} // namespace metal
} // namespace ctranslate2

#endif // CTRANSLATE2_METAL_SPEAKER_PROFILE_H_
