#include "gemm_cpu_accel.h"
#include <Accelerate/Accelerate.h>
#include <vector>
#include <dispatch/dispatch.h>

void gemm_cpu(const float* a, const float* b, float* c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

void gemm_accelerate(const float* a, const float* b, float* c, int m, int n, int k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}

void gemm_gcd(const float* a, const float* b, float* c, int m, int n, int k) {
    dispatch_apply(m, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    });
}

void gemm_accelerate_gcd(const std::vector<std::vector<float>>& a,
                         const std::vector<std::vector<float>>& b,
                         std::vector<std::vector<float>>& c,
                         int m, int n, int k, int batch) {
    dispatch_apply(batch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bidx) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a[bidx].data(), k, b[bidx].data(), n, 0.0f, c[bidx].data(), n);
    });
}
