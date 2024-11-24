#include <metal_stdlib>
using namespace metal;

// Basic compute operations
kernel void add_kernel(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* c [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
    c[index] = a[index] + b[index];
}

kernel void multiply_kernel(device const float* a [[buffer(0)]],
                          device const float* b [[buffer(1)]],
                          device float* c [[buffer(2)]],
                          uint index [[thread_position_in_grid]]) {
    c[index] = a[index] * b[index];
}

kernel void relu_kernel(device const float* x [[buffer(0)]],
                       device float* y [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    y[index] = max(0.0f, x[index]);
}

kernel void softmax_kernel(device const float* input [[buffer(0)]],
                         device float* output [[buffer(1)]],
                         constant uint& depth [[buffer(2)]],
                         uint batch_idx [[thread_position_in_grid]]) {
    const device float* batch_input = input + batch_idx * depth;
    device float* batch_output = output + batch_idx * depth;
    
    // Find max for numerical stability
    float max_val = batch_input[0];
    for (uint i = 1; i < depth; ++i) {
        max_val = max(max_val, batch_input[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < depth; ++i) {
        float val = exp(batch_input[i] - max_val);
        batch_output[i] = val;
        sum += val;
    }
    
    // Normalize
    for (uint i = 0; i < depth; ++i) {
        batch_output[i] /= sum;
    }
}

kernel void layer_norm_kernel(device const float* input [[buffer(0)]],
                            device const float* gamma [[buffer(1)]],
                            device const float* beta [[buffer(2)]],
                            device float* output [[buffer(3)]],
                            constant uint& hidden_size [[buffer(4)]],
                            constant float& epsilon [[buffer(5)]],
                            uint batch_idx [[thread_position_in_grid]]) {
    const device float* batch_input = input + batch_idx * hidden_size;
    device float* batch_output = output + batch_idx * hidden_size;
    
    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        mean += batch_input[i];
    }
    mean /= hidden_size;
    
    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        float diff = batch_input[i] - mean;
        variance += diff * diff;
    }
    variance /= hidden_size;
    
    // Normalize
    float inv_std = rsqrt(variance + epsilon);
    for (uint i = 0; i < hidden_size; ++i) {
        float normalized = (batch_input[i] - mean) * inv_std;
        batch_output[i] = gamma[i] * normalized + beta[i];
    }
}

// Matrix multiplication kernel (naive implementation - can be optimized further)
kernel void gemm_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* c [[buffer(2)]],
                       constant uint& M [[buffer(3)]],
                       constant uint& N [[buffer(4)]],
                       constant uint& K [[buffer(5)]],
                       constant float& alpha [[buffer(6)]],
                       constant float& beta [[buffer(7)]],
                       constant bool& a_trans [[buffer(8)]],
                       constant bool& b_trans [[buffer(9)]],
                       uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.x;
    uint col = pos.y;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        uint a_idx = a_trans ? (k * M + row) : (row * K + k);
        uint b_idx = b_trans ? (col * K + k) : (k * N + col);
        sum += a[a_idx] * b[b_idx];
    }
    
    uint c_idx = row * N + col;
    c[c_idx] = alpha * sum + beta * c[c_idx];
}
