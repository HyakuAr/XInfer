// Citing vendor documentation per AGENTS.md §5:
// - docs/vendor/xe-gpu-architecture.md (lines 48-50):
//   Vector Engine (VE) ALUs support native FP16 and FP32 operations.
// - docs/vendor/thread-mapping-occupancy.md (lines 9-15):
//   Sub-group size 16 maps to one Vector Engine hardware thread.

#include "elementwise.h"
#include <cmath>

namespace xinfer::ops {

// =============================================================================
// FP32 Elementwise Operations (Reference & Oracle Support)
// =============================================================================

sycl::event swiglu(sycl::queue& q, float* out, const float* gate, const float* up, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 4 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 4);
        const auto* gate_v = reinterpret_cast<const sycl::vec<float, 4>*>(gate);
        const auto* up_v   = reinterpret_cast<const sycl::vec<float, 4>*>(up);
        auto* out_v        = reinterpret_cast<sycl::vec<float, 4>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<float, 4> g = gate_v[i];
            sycl::vec<float, 4> u = up_v[i];
            sycl::vec<float, 4> res;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val = g[k];
                res[k] = (val / (1.0f + sycl::exp(-val))) * u[k];
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float g = gate[i];
        float silu_g = g / (1.0f + sycl::exp(-g));
        out[i] = silu_g * up[i];
    });
}

sycl::event silu(sycl::queue& q, float* out, const float* in, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 4 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 4);
        const auto* in_v = reinterpret_cast<const sycl::vec<float, 4>*>(in);
        auto* out_v      = reinterpret_cast<sycl::vec<float, 4>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<float, 4> x = in_v[i];
            sycl::vec<float, 4> res;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float val = x[k];
                res[k] = val / (1.0f + sycl::exp(-val));
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float x = in[i];
        out[i] = x / (1.0f + sycl::exp(-x));
    });
}

sycl::event add(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 4 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 4);
        const auto* a_v = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<float, 4>*>(b);
        auto* out_v     = reinterpret_cast<sycl::vec<float, 4>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            out_v[i] = a_v[i] + b_v[i];
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = a[i] + b[i];
    });
}

sycl::event add_inplace(sycl::queue& q, float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 4 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 4);
        auto* a_v       = reinterpret_cast<sycl::vec<float, 4>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<float, 4>*>(b);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            a_v[i] += b_v[i];
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        a[i] += b[i];
    });
}

sycl::event mul(sycl::queue& q, float* out, const float* a, const float* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 4 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 4);
        const auto* a_v = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<float, 4>*>(b);
        auto* out_v     = reinterpret_cast<sycl::vec<float, 4>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            out_v[i] = a_v[i] * b_v[i];
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = a[i] * b[i];
    });
}

// =============================================================================
// FP16 (sycl::half) Elementwise Operations (In-Kernel FP32 Accumulation)
// =============================================================================

sycl::event swiglu(sycl::queue& q, sycl::half* out, const sycl::half* gate, const sycl::half* up, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 8 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 8);
        const auto* gate_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(gate);
        const auto* up_v   = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(up);
        auto* out_v        = reinterpret_cast<sycl::vec<sycl::half, 8>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<sycl::half, 8> g = gate_v[i];
            sycl::vec<sycl::half, 8> u = up_v[i];
            sycl::vec<sycl::half, 8> res;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float val = static_cast<float>(g[k]);
                float u_val = static_cast<float>(u[k]);
                res[k] = static_cast<sycl::half>((val / (1.0f + sycl::exp(-val))) * u_val);
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float g = static_cast<float>(gate[i]);
        float silu_g = g / (1.0f + sycl::exp(-g));
        out[i] = static_cast<sycl::half>(silu_g * static_cast<float>(up[i]));
    });
}

sycl::event silu(sycl::queue& q, sycl::half* out, const sycl::half* in, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 8 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 8);
        const auto* in_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(in);
        auto* out_v      = reinterpret_cast<sycl::vec<sycl::half, 8>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<sycl::half, 8> x = in_v[i];
            sycl::vec<sycl::half, 8> res;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float val = static_cast<float>(x[k]);
                res[k] = static_cast<sycl::half>(val / (1.0f + sycl::exp(-val)));
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        float x = static_cast<float>(in[i]);
        out[i] = static_cast<sycl::half>(x / (1.0f + sycl::exp(-x)));
    });
}

sycl::event add(sycl::queue& q, sycl::half* out, const sycl::half* a, const sycl::half* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 8 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 8);
        const auto* a_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(b);
        auto* out_v     = reinterpret_cast<sycl::vec<sycl::half, 8>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<sycl::half, 8> av = a_v[i];
            sycl::vec<sycl::half, 8> bv = b_v[i];
            sycl::vec<sycl::half, 8> res;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                res[k] = static_cast<sycl::half>(static_cast<float>(av[k]) + static_cast<float>(bv[k]));
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = static_cast<sycl::half>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    });
}

sycl::event add_inplace(sycl::queue& q, sycl::half* a, const sycl::half* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 8 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 8);
        auto* a_v       = reinterpret_cast<sycl::vec<sycl::half, 8>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(b);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<sycl::half, 8> av = a_v[i];
            sycl::vec<sycl::half, 8> bv = b_v[i];
            sycl::vec<sycl::half, 8> res;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                res[k] = static_cast<sycl::half>(static_cast<float>(av[k]) + static_cast<float>(bv[k]));
            }
            a_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        a[i] = static_cast<sycl::half>(static_cast<float>(a[i]) + static_cast<float>(b[i]));
    });
}

sycl::event mul(sycl::queue& q, sycl::half* out, const sycl::half* a, const sycl::half* b, int64_t num_elements) {
    if (num_elements <= 0) return sycl::event{};
    if (num_elements % 8 == 0) {
        size_t n_vec = static_cast<size_t>(num_elements / 8);
        const auto* a_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(a);
        const auto* b_v = reinterpret_cast<const sycl::vec<sycl::half, 8>*>(b);
        auto* out_v     = reinterpret_cast<sycl::vec<sycl::half, 8>*>(out);

        return q.parallel_for(sycl::range<1>(n_vec), [=](sycl::id<1> idx) {
            size_t i = idx[0];
            sycl::vec<sycl::half, 8> av = a_v[i];
            sycl::vec<sycl::half, 8> bv = b_v[i];
            sycl::vec<sycl::half, 8> res;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                res[k] = static_cast<sycl::half>(static_cast<float>(av[k]) * static_cast<float>(bv[k]));
            }
            out_v[i] = res;
        });
    }
    return q.parallel_for(sycl::range<1>(static_cast<size_t>(num_elements)), [=](sycl::id<1> idx) {
        int64_t i = idx[0];
        out[i] = static_cast<sycl::half>(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    });
}

} // namespace xinfer::ops
