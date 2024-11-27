#ifndef CTRANSLATE2_METAL_SPEAKER_MANAGER_H_
#define CTRANSLATE2_METAL_SPEAKER_MANAGER_H_

#include "speaker_profile.h"
#include <unordered_map>
#include <mutex>
#include <optional>

namespace ctranslate2 {
namespace metal {

class SpeakerManager {
public:
    SpeakerManager(id<MTLDevice> device, id<MTLCommandQueue> command_queue);
    ~SpeakerManager();
    
    // Speaker identification
    std::string identify_speaker(const float* audio, size_t size);
    
    // Profile management
    bool save_speaker_model(const std::string& speaker_id);
    bool load_speaker_model(const std::string& speaker_id);
    void create_new_speaker_profile(const std::string& speaker_id);
    
    // Audio processing
    void process_audio_with_speaker_detection(const float* audio, size_t size);
    
    // Current speaker info
    std::string get_current_speaker_id() const { return _current_speaker_id; }
    std::optional<SpeakerProfile> get_current_profile() const;
    
private:
    static constexpr size_t kDetectionIntervalSamples = 44100 * 2; // 2 seconds
    
    // Hardware handles
    id<MTLDevice> _device;
    id<MTLCommandQueue> _command_queue;
    
    // Neural engine compute pipeline for fingerprint extraction
    MPSNNGraph* _fingerprint_pipeline;
    
    // Speaker profiles
    std::unordered_map<std::string, SpeakerProfile> _active_profiles;
    std::string _current_speaker_id;
    mutable std::mutex _profiles_mutex;
    
    // Internal methods
    VoiceFingerprint extract_fingerprint(const float* audio, size_t size);
    std::string match_fingerprint(const VoiceFingerprint& fp);
    void process_with_model(const float* audio, size_t size, SpeakerProfile& profile);
    void setup_neural_engine_pipeline();
    
    // File operations
    static std::string get_model_path(const std::string& speaker_id);
    static std::vector<uint8_t> compress_model(const std::string& model_json);
    static std::string decompress_model(std::istream& compressed_data);
};

} // namespace metal
} // namespace ctranslate2

#endif // CTRANSLATE2_METAL_SPEAKER_MANAGER_H_
