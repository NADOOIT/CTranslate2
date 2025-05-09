#include <metal_stdlib>
#include <metal_compute>

using namespace metal;

struct AttentionParams {
    uint32_t batch_size;
    uint32_t num_heads;
    uint32_t queries_per_head;
    uint32_t keys_per_head;
    uint32_t dim_per_head;
    float scale;
    uint32_t is_causal;
};

// Basic compute kernels for CTranslate2
kernel void add_kernel(
    device const float* input1 [[buffer(0)]],
    device const float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        output[tid] = alpha * input1[tid] + beta * input2[tid];
    }
}

kernel void multiply_kernel(
    device const float* input1 [[buffer(0)]],
    device const float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        output[tid] = alpha * input1[tid] * input2[tid] + beta;
    }
}

kernel void relu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        output[tid] = fmax(0.0f, input[tid]);
    }
}

kernel void gelu_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < size) {
        float x = input[tid];
        float cdf = 0.5f * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
        output[tid] = x * cdf;
    }
}

kernel void softmax_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& feature_size [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint batch_idx = tid.x;
    if (batch_idx < batch_size) {
        const device float* batch_input = input + batch_idx * feature_size;
        device float* batch_output = output + batch_idx * feature_size;
        
        // Find max value for numerical stability
        float max_val = batch_input[0];
        for (uint i = 1; i < feature_size; ++i) {
            max_val = fmax(max_val, batch_input[i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (uint i = 0; i < feature_size; ++i) {
            float val = exp(batch_input[i] - max_val);
            batch_output[i] = val;
            sum += val;
        }
        
        // Normalize
        for (uint i = 0; i < feature_size; ++i) {
            batch_output[i] /= sum;
        }
    }
}

kernel void gemm_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    constant bool& transpose_a [[buffer(8)]],
    constant bool& transpose_b [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint row = gid.x;
    const uint col = gid.y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (uint k = 0; k < K; ++k) {
            uint a_idx = transpose_a ? (k * M + row) : (row * K + k);
            uint b_idx = transpose_b ? (col * K + k) : (k * N + col);
            sum += a[a_idx] * b[b_idx];
        }
        c[row * N + col] = alpha * sum + beta * c[row * N + col];
    }
}

kernel void attention_kernel(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]]
) {
    // Get indices
    const uint query_pos = tid.x;    // Position in the query sequence
    const uint batch_idx = tid.y;    // Batch index
    
    // Check bounds
    if (query_pos >= params.queries_per_head || batch_idx >= params.batch_size) {
        return;
    }
    
    // Constants for better readability
    const uint H = params.num_heads;
    const uint Q = params.queries_per_head;
    const uint K = params.keys_per_head;
    const uint D = params.dim_per_head;
    
    // Compute attention scores for this query position across all heads
    for (uint head = 0; head < H; head++) {
        // Base indices for this batch, head, and query position
        const uint q_base = (batch_idx * H * Q * D) + (head * Q * D) + (query_pos * D);
        const uint k_base = (batch_idx * H * K * D) + (head * K * D);
        const uint qk_base = (batch_idx * H * Q * K) + (head * Q * K) + (query_pos * K);
        const uint v_base = (batch_idx * H * K * D) + (head * K * D);
        const uint out_base = (batch_idx * H * Q * D) + (head * Q * D) + (query_pos * D);
        
        // Compute attention scores
        threadgroup float scores[1024];  // Adjust size based on maximum sequence length
        for (uint k_pos = 0; k_pos < K; k_pos++) {
            float score = 0.0f;
            
            // Compute dot product between query and key
            for (uint d = 0; d < D; d++) {
                score += query[q_base + d] * key[k_base + k_pos * D + d];
            }
            
            // Apply scaling
            score *= params.scale;
            
            // Apply causal mask if needed
            if (params.is_causal && k_pos > query_pos) {
                score = -INFINITY;
            }
            
            // Apply attention mask if provided
            if (mask != nullptr) {
                score += mask[qk_base + k_pos];
            }
            
            scores[k_pos] = score;
        }
        
        // Apply softmax
        float max_score = -INFINITY;
        for (uint k_pos = 0; k_pos < K; k_pos++) {
            max_score = max(max_score, scores[k_pos]);
        }
        
        float sum_exp = 0.0f;
        for (uint k_pos = 0; k_pos < K; k_pos++) {
            scores[k_pos] = exp(scores[k_pos] - max_score);
            sum_exp += scores[k_pos];
        }
        
        for (uint k_pos = 0; k_pos < K; k_pos++) {
            scores[k_pos] /= sum_exp;
        }
        
        // Compute weighted sum of values
        for (uint d = 0; d < D; d++) {
            float weighted_sum = 0.0f;
            for (uint k_pos = 0; k_pos < K; k_pos++) {
                weighted_sum += scores[k_pos] * value[v_base + k_pos * D + d];
            }
            output[out_base + d] = weighted_sum;
        }
    }
}
