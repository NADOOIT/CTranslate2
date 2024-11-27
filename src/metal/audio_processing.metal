#include <metal_stdlib>
using namespace metal;

// Constants for 2nd order Butterworth filter coefficients
constant float a0 = 1.0f;
constant float a1 = -1.414213562f;  // -2 * cos(pi/4)
constant float a2 = 1.0f;
constant float b0 = 0.292893219f;   // 1 / (2 + sqrt(2))
constant float b1 = 0.585786438f;   // 2 * b0
constant float b2 = b0;

// State structure for filter
struct FilterState {
    float4 x1;  // Previous input sample
    float4 x2;  // Second previous input sample
    float4 y1;  // Previous output sample
    float4 y2;  // Second previous output sample
};

// Apply a 2nd order Butterworth filter
static float4 apply_filter(float4 input, thread FilterState& state, float cutoff) {
    // Scale coefficients based on cutoff frequency
    float scale = cutoff * cutoff;
    float4 scaled_b0 = float4(b0 * scale);
    float4 scaled_b1 = float4(b1 * scale);
    float4 scaled_b2 = float4(b2 * scale);
    float4 scaled_a1 = float4(a1 * cutoff);
    float4 scaled_a2 = float4(a2);
    
    // Apply difference equation
    float4 output = scaled_b0 * input + 
                   scaled_b1 * state.x1 + 
                   scaled_b2 * state.x2 - 
                   scaled_a1 * state.y1 - 
                   scaled_a2 * state.y2;
    
    // Update state
    state.x2 = state.x1;
    state.x1 = input;
    state.y2 = state.y1;
    state.y1 = output;
    
    return output;
}

kernel void process_audio_simd(device const float4* input [[buffer(0)]],
                             device float4* output [[buffer(1)]],
                             device const float* params [[buffer(2)]],
                             device FilterState* filter_states [[buffer(3)]],
                             uint index [[thread_position_in_grid]]) {
    // Load parameters
    float lowpass_cutoff = params[0];  // Normalized cutoff frequency (0-1)
    float highpass_cutoff = params[1]; // Normalized cutoff frequency (0-1)
    
    // Load input vector
    float4 in_vec = input[index];
    
    // Load filter state
    thread FilterState lowpass_state = filter_states[index * 2];
    thread FilterState highpass_state = filter_states[index * 2 + 1];
    
    // Apply filters in sequence
    float4 filtered = in_vec;
    filtered = apply_filter(filtered, lowpass_state, lowpass_cutoff);
    filtered = apply_filter(filtered, highpass_state, highpass_cutoff);
    
    // Store updated filter states
    filter_states[index * 2] = lowpass_state;
    filter_states[index * 2 + 1] = highpass_state;
    
    // Store result
    output[index] = filtered;
}
