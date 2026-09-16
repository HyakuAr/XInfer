#include "elementwise.h"
#include <cmath>

namespace xinfer::ops {

void swiglu(sycl::queue& q, float* out, const float* gate, const float* up, int64_t num_elements) {
    if (num_elements <= 0) return;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float g = gate[i];
        float silu_g = g / (1.0f + sycl::exp(-g));
        out[i] = silu_g * up[i];
    });
}

void silu(sycl::queue& q, float* out, const float* in, int64_t num_elements) {
    if (num_elements <= 0) return;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float x = in[i];
        out[i] = x / (1.0f + sycl::exp(-x));
    });
}

void add(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = a[i] + b[i];
    });
}

void add_inplace(sycl::queue& q, float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        a[i] += b[i];
    });
}

void mul(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return;
    q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = a[i] * b[i];
    });
}

} // namespace xinfer::ops
