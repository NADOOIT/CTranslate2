#include <metal_stdlib>
using namespace metal;

// Common parameter structures
struct GEMMParams {
    uint M;
    uint N;
    uint K;
    float alpha;
    float beta;
    uint lda;
    uint ldb;
    uint ldc;
    uint transpose_a;
    uint transpose_b;
};

struct ElementwiseParams {
    uint size;
    float alpha;
    float beta;
};

// Helper functions for float and half precision
template<typename T>
inline T get_element(const device T* matrix,
                    uint row,
                    uint col,
                    uint ld,
                    bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// GEMM kernel with template for float and half precision
template<typename T>
kernel void gemm_kernel_t(const device T* a [[buffer(0)]],
                         const device T* b [[buffer(1)]],
                         device T* c [[buffer(2)]],
                         constant GEMMParams& params [[buffer(3)]],
                         uint2 thread_position_in_grid [[thread_position_in_grid]],
                         uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                         uint2 threadgroup_position [[threadgroup_position]]) {
    constexpr int TILE_SIZE = 16;
    
    threadgroup T a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup T b_tile[TILE_SIZE][TILE_SIZE];
    
    const uint row = thread_position_in_grid.y;
    const uint col = thread_position_in_grid.x;
    
    if (row >= params.M || col >= params.N) {
        return;
    }
    
    T acc = 0.0;
    const uint num_tiles = (params.K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint tile = 0; tile < num_tiles; ++tile) {
        const uint tile_col = thread_position_in_threadgroup.x;
        const uint tile_row = thread_position_in_threadgroup.y;
        
        if (tile * TILE_SIZE + tile_col < params.K && row < params.M) {
            a_tile[tile_row][tile_col] = get_element(a,
                                                   row,
                                                   tile * TILE_SIZE + tile_col,
                                                   params.lda,
                                                   params.transpose_a);
        } else {
            a_tile[tile_row][tile_col] = 0.0;
        }
        
        if (tile * TILE_SIZE + tile_row < params.K && col < params.N) {
            b_tile[tile_row][tile_col] = get_element(b,
                                                   tile * TILE_SIZE + tile_row,
                                                   col,
                                                   params.ldb,
                                                   params.transpose_b);
        } else {
            b_tile[tile_row][tile_col] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; ++k) {
            acc += a_tile[tile_row][k] * b_tile[k][tile_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < params.M && col < params.N) {
        const uint c_idx = row * params.ldc + col;
        c[c_idx] = T(params.alpha) * acc + T(params.beta) * c[c_idx];
    }
}

// Explicit instantiations for float and half precision GEMM
kernel void gemm_kernel_fp32(const device float* a [[buffer(0)]],
                           const device float* b [[buffer(1)]],
                           device float* c [[buffer(2)]],
                           constant GEMMParams& params [[buffer(3)]],
                           uint2 thread_position_in_grid [[thread_position_in_grid]],
                           uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                           uint2 threadgroup_position [[threadgroup_position]]) {
    gemm_kernel_t<float>(a, b, c, params, thread_position_in_grid, thread_position_in_threadgroup, threadgroup_position);
}

kernel void gemm_kernel_fp16(const device half* a [[buffer(0)]],
                           const device half* b [[buffer(1)]],
                           device half* c [[buffer(2)]],
                           constant GEMMParams& params [[buffer(3)]],
                           uint2 thread_position_in_grid [[thread_position_in_grid]],
                           uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                           uint2 threadgroup_position [[threadgroup_position]]) {
    gemm_kernel_t<half>(a, b, c, params, thread_position_in_grid, thread_position_in_threadgroup, threadgroup_position);
}

// ReLU activation function
kernel void relu_kernel(device float* output [[buffer(0)]],
                       const device float* input [[buffer(1)]],
                       constant ElementwiseParams& params [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= params.size) return;
    output[index] = max(0.0f, input[index]);
}

// Gelu activation function
kernel void gelu_kernel(device float* output [[buffer(0)]],
                       const device float* input [[buffer(1)]],
                       constant ElementwiseParams& params [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= params.size) return;
    float x = input[index];
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_pi = 0.7978845608028654f;  // sqrt(2/π)
    float cube = x * x * x;
    float inner = sqrt_2_pi * (x + 0.044715f * cube);
    output[index] = x * 0.5f * (1.0f + tanh(inner));
}

// Element-wise addition with optional scaling
kernel void add_kernel(device float* output [[buffer(0)]],
                      const device float* input1 [[buffer(1)]],
                      const device float* input2 [[buffer(2)]],
                      constant ElementwiseParams& params [[buffer(3)]],
                      uint index [[thread_position_in_grid]]) {
    if (index >= params.size) return;
    output[index] = params.alpha * input1[index] + params.beta * input2[index];
}

// Element-wise multiplication with optional scaling
kernel void multiply_kernel(device float* output [[buffer(0)]],
                          const device float* input1 [[buffer(1)]],
                          const device float* input2 [[buffer(2)]],
                          constant ElementwiseParams& params [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
    if (index >= params.size) return;
    output[index] = params.alpha * input1[index] * input2[index] + params.beta * output[index];
}

// Layer normalization
kernel void layer_norm_kernel(device float* output [[buffer(0)]],
                            const device float* input [[buffer(1)]],
                            const device float* gamma [[buffer(2)]],
                            const device float* beta [[buffer(3)]],
                            constant uint& hidden_size [[buffer(4)]],
                            uint index [[thread_position_in_grid]]) {
    const uint batch_idx = index / hidden_size;
    const uint offset = batch_idx * hidden_size;
    
    // First pass: compute mean
    float sum = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        sum += input[offset + i];
    }
    float mean = sum / float(hidden_size);
    
    // Second pass: compute variance
    float variance = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        float diff = input[offset + i] - mean;
        variance += diff * diff;
    }
    variance /= float(hidden_size);
    
    // Normalize and apply scale and shift
    const uint curr_pos = index % hidden_size;
    float normalized = (input[index] - mean) / sqrt(variance + 1e-5f);
    output[index] = gamma[curr_pos] * normalized + beta[curr_pos];
}

// Softmax parameters
struct SoftmaxParams {
    uint size;
    uint batch_size;
    uint feature_size;
};

// Attention parameters
struct AttentionParams {
    uint batch_size;
    uint num_heads;
    uint head_dim;
    uint seq_len;
    float scale;
};

// Softmax implementation
kernel void softmax_kernel(device float* output [[buffer(0)]],
                         const device float* input [[buffer(1)]],
                         constant SoftmaxParams& params [[buffer(2)]],
                         uint index [[thread_position_in_grid]]) {
    if (index >= params.batch_size) return;
    
    // Compute max for numerical stability
    float max_val = input[index * params.feature_size];
    for (uint i = 1; i < params.feature_size; ++i) {
        max_val = max(max_val, input[index * params.feature_size + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < params.feature_size; ++i) {
        const uint idx = index * params.feature_size + i;
        output[idx] = exp(input[idx] - max_val);
        sum += output[idx];
    }
    
    // Normalize
    const float inv_sum = 1.0f / sum;
    for (uint i = 0; i < params.feature_size; ++i) {
        output[index * params.feature_size + i] *= inv_sum;
    }
}

// Scaled dot-product attention
kernel void attention_kernel(device float* output [[buffer(0)]],
                           const device float* query [[buffer(1)]],
                           const device float* key [[buffer(2)]],
                           const device float* value [[buffer(3)]],
                           constant AttentionParams& params [[buffer(4)]],
                           threadgroup float* shared_scores [[threadgroup(0)]],
                           uint3 thread_position_in_grid [[thread_position_in_grid]],
                           uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                           uint3 threadgroup_position [[threadgroup_position]]) {
    const uint b = thread_position_in_grid.x;  // batch index
    const uint h = thread_position_in_grid.y;  // head index
    const uint i = thread_position_in_grid.z;  // query sequence index
    
    if (b >= params.batch_size || h >= params.num_heads || i >= params.seq_len) return;
    
    const uint head_size = params.head_dim;
    const uint seq_len = params.seq_len;
    
    // Compute attention scores for one query position
    const uint q_offset = ((b * params.num_heads + h) * seq_len + i) * head_size;
    float scores[32];  // Assuming max sequence length is 32, adjust if needed
    
    // Compute attention scores
    for (uint j = 0; j < seq_len; ++j) {
        const uint k_offset = ((b * params.num_heads + h) * seq_len + j) * head_size;
        float score = 0.0f;
        
        // Compute dot product
        for (uint d = 0; d < head_size; ++d) {
            score += query[q_offset + d] * key[k_offset + d];
        }
        
        scores[j] = score * params.scale;
    }
    
    // Apply softmax to attention scores
    float max_score = scores[0];
    for (uint j = 1; j < seq_len; ++j) {
        max_score = max(max_score, scores[j]);
    }
    
    float sum = 0.0f;
    for (uint j = 0; j < seq_len; ++j) {
        scores[j] = exp(scores[j] - max_score);
        sum += scores[j];
    }
    
    const float inv_sum = 1.0f / sum;
    for (uint j = 0; j < seq_len; ++j) {
        scores[j] *= inv_sum;
    }
    
    // Compute weighted sum of values
    const uint o_offset = ((b * params.num_heads + h) * seq_len + i) * head_size;
    for (uint d = 0; d < head_size; ++d) {
        float weighted_sum = 0.0f;
        for (uint j = 0; j < seq_len; ++j) {
            const uint v_offset = ((b * params.num_heads + h) * seq_len + j) * head_size;
            weighted_sum += scores[j] * value[v_offset + d];
        }
        output[o_offset + d] = weighted_sum;
    }
}

// Flash Attention implementation for better memory efficiency
kernel void flash_attention_kernel(device float* output [[buffer(0)]],
                                 const device float* query [[buffer(1)]],
                                 const device float* key [[buffer(2)]],
                                 const device float* value [[buffer(3)]],
                                 constant AttentionParams& params [[buffer(4)]],
                                 threadgroup float* shared_kv [[threadgroup(0)]],
                                 uint3 thread_position_in_grid [[thread_position_in_grid]],
                                 uint3 threadgroup_position [[threadgroup_position]]) {
    constexpr uint BLOCK_SIZE = 32;  // Adjust based on hardware capabilities
    const uint b = thread_position_in_grid.x;  // batch index
    const uint h = thread_position_in_grid.y;  // head index
    const uint i = thread_position_in_grid.z;  // query sequence index
    
    if (b >= params.batch_size || h >= params.num_heads || i >= params.seq_len) return;
    
    const uint head_size = params.head_dim;
    const uint seq_len = params.seq_len;
    
    // Initialize accumulators
    float acc[32];  // Assuming head_dim <= 32, adjust if needed
    float max_score = -INFINITY;
    float sum = 0.0f;
    
    for (uint d = 0; d < head_size; ++d) {
        acc[d] = 0.0f;
    }
    
    // Process key-value pairs in blocks
    for (uint block_start = 0; block_start < seq_len; block_start += BLOCK_SIZE) {
        const uint block_end = min(block_start + BLOCK_SIZE, seq_len);
        
        // Load query block
        const uint q_offset = ((b * params.num_heads + h) * seq_len + i) * head_size;
        float q_block[32];  // Assuming head_dim <= 32
        for (uint d = 0; d < head_size; ++d) {
            q_block[d] = query[q_offset + d];
        }
        
        // Process key-value pairs in current block
        for (uint j = block_start; j < block_end; ++j) {
            const uint k_offset = ((b * params.num_heads + h) * seq_len + j) * head_size;
            const uint v_offset = k_offset;
            
            // Compute attention score
            float score = 0.0f;
            for (uint d = 0; d < head_size; ++d) {
                score += q_block[d] * key[k_offset + d];
            }
            score *= params.scale;
            
            // Update running maximum
            const float old_max_score = max_score;
            max_score = max(max_score, score);
            
            // Update accumulators
            const float scale = exp(old_max_score - max_score);
            sum *= scale;
            
            const float new_weight = exp(score - max_score);
            sum += new_weight;
            
            for (uint d = 0; d < head_size; ++d) {
                acc[d] = acc[d] * scale + new_weight * value[v_offset + d];
            }
        }
    }
    
    // Write final output
    const uint o_offset = ((b * params.num_heads + h) * seq_len + i) * head_size;
    const float inv_sum = 1.0f / sum;
    for (uint d = 0; d < head_size; ++d) {
        output[o_offset + d] = acc[d] * inv_sum;
    }
}
