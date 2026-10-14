// Four complex atoms per AVX-512 vector, retaining the four-atom working set.
// The low four lanes are real components, the high four are imaginary.
#pragma once
#include <immintrin.h>
#include <limits>
#if !defined(__AVX512F__) || !defined(__AVX512DQ__) || !defined(__AVX512VL__)
#error "Packed complex densities require AVX-512F, DQ and VL"
#endif
static_assert(WBC_DENSITY_BLOCK == 4, "Packed complex densities use four atoms");

namespace wbc_packed {
struct alignas(64) Complex { __m512d v; };
inline __m512d real_twice(__m512d x) { return _mm512_shuffle_f64x2(x, x, 0x44); }
inline __m512d imag_twice(__m512d x) { return _mm512_shuffle_f64x2(x, x, 0xee); }
inline __m512d swap(__m512d x) { return _mm512_shuffle_f64x2(x, x, 0x4e); }
inline __m512d multiply(__m512d a, __m512d b) {
  const __m512d first = _mm512_mul_pd(real_twice(a), b);
  const __m512d second = _mm512_mul_pd(imag_twice(a), swap(b));
  const __m512d sign = _mm512_maskz_mov_pd(0x0f, _mm512_set1_pd(-0.0));
  return _mm512_add_pd(first, _mm512_xor_pd(second, sign));
}
inline __m512d adjoint(__m512d q, __m512d factor) {
  const __m512d first = _mm512_mul_pd(q, real_twice(factor));
  const __m512d second = _mm512_mul_pd(swap(q), imag_twice(factor));
  const __m512d sign = _mm512_maskz_mov_pd(0xf0, _mm512_set1_pd(-0.0));
  return _mm512_add_pd(first, _mm512_xor_pd(second, sign));
}
// Terminal products have structurally real incoming adjoints. Avoid the
// imaginary-zero products and permutations without changing their algebra.
inline __m512d real_adjoint(__m512d q, __m512d factor) {
  const __m512d sign = _mm512_maskz_mov_pd(0xf0, _mm512_set1_pd(-0.0));
  return _mm512_xor_pd(_mm512_mul_pd(real_twice(q), factor), sign);
}
inline __m512d load(const double* real, const double* imag, __mmask8 mask) {
  const __m256d r = _mm256_maskz_loadu_pd(mask, real);
  const __m256d i = _mm256_maskz_loadu_pd(mask, imag);
  return _mm512_insertf64x4(_mm512_castpd256_pd512(r), i, 1);
}
inline __m512d real_only(const double* real) { return _mm512_maskz_loadu_pd(0x0f, real); }
}

template <int mode, int fixed_density = 0>
void PackedComplexDensity(
    const int64_t nf, const int64_t na, const int64_t np,
    const int64_t nt, const int64_t dynamic_density, const int32* left,
    const int32* right, const int32* terms,
    const std::vector<unsigned char>& is_parent,
    const double* ar, const double* ai, const double* weights,
    const double* seed, const double* hr, const double* hi, const double* hw,
    double* o0, double* o1, double* ow, double* os,
    const bool zero_weight_direction, OpKernelContext* context) {
  if constexpr (fixed_density == 0) {
    if (dynamic_density == 2) {
      // The two ACE embedding channels remain independent. This only makes
      // their loop extent visible to the compiler, with the same sum order.
      PackedComplexDensity<mode, 2>(nf, na, np, nt, dynamic_density, left, right,
          terms, is_parent, ar, ai, weights, seed, hr, hi, hw, o0, o1, ow, os,
          zero_weight_direction, context);
      return;
    }
  }
  using namespace wbc_packed;
  constexpr int block = 4;
  const int64_t nd = fixed_density ? fixed_density : dynamic_density;
  const int64_t nn = nf + np, nw = nt * nd, nblocks = (na + block - 1) / block;
  if (nn > std::numeric_limits<int32>::max()) {
    // Keep a correct fallback for a plan beyond the compact index range.
    CompactDensity<mode>(nf, na, np, nt, nd, left, right, terms, is_parent,
        ar, ai, weights, seed, hr, hi, hw, o0, o1, ow, os, zero_weight_direction, context);
    return;
  }
  const bool reverse = mode >= 1 && (o0 || o1);
  std::vector<int32> slots(nn, -1), term_ids(nn, -1), lhs(np), rhs(np);
  int64_t ns = nf;
  for (int64_t f = 0; f < nf; ++f) slots[f] = f;
  for (int64_t k = 0; k < np; ++k) {
    if (is_parent[nf + k]) slots[nf + k] = ns++;
    lhs[k] = slots[left[k]];
    rhs[k] = slots[right[k]];
  }
  for (int64_t t = 0; t < nt; ++t) term_ids[terms[t]] = t;
  const auto* workers = context->device()->tensorflow_cpu_worker_threads();
  const int64_t nchunks = std::min<int64_t>(nblocks, std::max(1, workers->num_threads));
  auto dispatch = [&](auto weight_tag, auto zero_tag) {
    constexpr bool compute_weights = decltype(weight_tag)::value;
    constexpr bool zero_hw = decltype(zero_tag)::value;
    std::vector<double> partial_weights(compute_weights ? nchunks * nw : 0, 0.);
    auto work = [&](int64_t begin, int64_t end) {
      std::vector<Complex> p(ns), dp(mode == 2 ? ns : 0);
      std::vector<Complex> g(reverse ? ns : 0), dg(mode == 2 && reverse ? ns : 0);
      std::vector<double> rho(nd * block), s(nd * block);
      const Complex zero{_mm512_setzero_pd()};
      for (int64_t chunk = begin; chunk < end; ++chunk) {
        double* gw = compute_weights ? partial_weights.data() + chunk * nw : nullptr;
        for (int64_t atom_block = nblocks * chunk / nchunks;
             atom_block < nblocks * (chunk + 1) / nchunks; ++atom_block) {
          const int64_t first = atom_block * block;
          const int valid = std::min<int64_t>(block, na - first);
          const __mmask8 mask = static_cast<__mmask8>((1u << valid) - 1);
          for (int64_t f = 0; f < nf; ++f) {
            const int64_t src = f * na + first;
            p[f].v = load(ar + src, ai + src, mask);
            if constexpr (mode == 2) dp[f].v = load(hr + src, hi + src, mask);
          }
          for (int64_t k = 0; k < np; ++k) {
            const int64_t n = slots[nf + k], l = lhs[k], r = rhs[k];
            if (n < 0) continue;
            p[n].v = multiply(p[l].v, p[r].v);
            if constexpr (mode == 2)
              dp[n].v = _mm512_add_pd(multiply(dp[l].v, p[r].v), multiply(p[l].v, dp[r].v));
          }
          std::fill(rho.begin(), rho.end(), 0.);
          std::fill(g.begin(), g.end(), zero);
          std::fill(dg.begin(), dg.end(), zero);
          if constexpr (mode >= 1)
            for (int64_t d = 0; d < nd; ++d)
              _mm256_storeu_pd(s.data() + d * block,
                  _mm256_maskz_loadu_pd(mask, seed + d * na + first));
          for (int64_t t = 0; t < nt; ++t) {
            const int64_t node = slots[terms[t]];
            const bool need_value = mode == 0 || (mode == 1 && compute_weights) ||
                                    (mode == 2 && !zero_hw && os);
            const bool need_tangent = mode == 2 && (compute_weights || os);
            double value[block]{}, tangent[block]{};
            if (node >= 0) {
              if (need_value) _mm256_storeu_pd(value, _mm512_castpd512_pd256(p[node].v));
              if (need_tangent) _mm256_storeu_pd(tangent, _mm512_castpd512_pd256(dp[node].v));
            } else {
              const int64_t k = terms[t] - nf, l = lhs[k], r = rhs[k];
              if (need_value) _mm256_storeu_pd(value,
                  _mm512_castpd512_pd256(multiply(p[l].v, p[r].v)));
              if (need_tangent) _mm256_storeu_pd(tangent, _mm512_castpd512_pd256(
                  _mm512_add_pd(multiply(dp[l].v, p[r].v), multiply(p[l].v, dp[r].v))));
            }
            for (int64_t d = 0; d < nd; ++d) {
              const int64_t wd = t * nd + d, ds = d * block;
              const double coefficient = weights[wd];
              if constexpr (mode == 0) {
                for (int lane = 0; lane < block; ++lane)
                  rho[ds + lane] += coefficient * value[lane];
              } else {
                if (reverse && node >= 0) {
                  const __m512d incoming = real_only(s.data() + ds);
                  g[node].v = _mm512_add_pd(g[node].v, _mm512_mul_pd(_mm512_set1_pd(coefficient), incoming));
                  if constexpr (mode == 2 && !zero_hw)
                    dg[node].v = _mm512_add_pd(dg[node].v, _mm512_mul_pd(_mm512_set1_pd(hw[wd]), incoming));
                }
                if constexpr (mode == 2) {
                  if (os) for (int lane = 0; lane < block; ++lane) {
                    if constexpr (zero_hw) rho[ds + lane] += coefficient * tangent[lane];
                    else rho[ds + lane] += coefficient * tangent[lane] + hw[wd] * value[lane];
                  }
                }
                if constexpr (compute_weights)
                  for (int lane = 0; lane < valid; ++lane) {
                    if constexpr (mode == 1) gw[wd] += value[lane] * s[ds + lane];
                    else gw[wd] += tangent[lane] * s[ds + lane];
                  }
              }
            }
          }
          if constexpr (mode == 0 || mode == 2) {
            double* target = mode == 0 ? o0 : os;
            if (target) for (int64_t d = 0; d < nd; ++d)
              _mm256_mask_storeu_pd(target + d * na + first, mask, _mm256_loadu_pd(rho.data() + d * block));
          }
          if constexpr (mode >= 1) {
            if (reverse) {
              for (int64_t k = np; k-- > 0;) {
                const int64_t node = slots[nf + k], l = lhs[k], r = rhs[k];
                __m512d q, dq = _mm512_setzero_pd();
                if (node >= 0) {
                  q = g[node].v;
                  if constexpr (mode == 2) dq = dg[node].v;
                } else {
                  const int64_t t = term_ids[nf + k];
                  if (t < 0) continue;
                  double real[block]{}, dre[block]{};
                  for (int64_t d = 0; d < nd; ++d)
                    for (int lane = 0; lane < block; ++lane) {
                      real[lane] += weights[t * nd + d] * s[d * block + lane];
                      if constexpr (mode == 2 && !zero_hw)
                        dre[lane] += hw[t * nd + d] * s[d * block + lane];
                    }
                  q = real_only(real);
                  if constexpr (mode == 2) dq = real_only(dre);
                }
                auto propagate = [&](auto terminal_tag) {
                  constexpr bool terminal = decltype(terminal_tag)::value;
                  auto contract = [&](__m512d incoming, __m512d factor) {
                    if constexpr (terminal) return real_adjoint(incoming, factor);
                    else return adjoint(incoming, factor);
                  };
                  if constexpr (mode == 2) {
                    if constexpr (terminal && zero_hw) {
                      // dq is identically zero here, not just currently small.
                      dg[l].v = _mm512_add_pd(dg[l].v, contract(q, dp[r].v));
                      dg[r].v = _mm512_add_pd(dg[r].v, contract(q, dp[l].v));
                    } else {
                      dg[l].v = _mm512_add_pd(dg[l].v, _mm512_add_pd(contract(dq, p[r].v), contract(q, dp[r].v)));
                      dg[r].v = _mm512_add_pd(dg[r].v, _mm512_add_pd(contract(dq, p[l].v), contract(q, dp[l].v)));
                    }
                  }
                  // Sequential stores also handle repeated factors (l == r).
                  g[l].v = _mm512_add_pd(g[l].v, contract(q, p[r].v));
                  g[r].v = _mm512_add_pd(g[r].v, contract(q, p[l].v));
                };
                if (node < 0) propagate(std::true_type{});
                else propagate(std::false_type{});
              }
              for (int64_t f = 0; f < nf; ++f) {
                const __m512d answer = mode == 1 ? g[f].v : dg[f].v;
                if (o0) _mm256_mask_storeu_pd(o0 + f * na + first, mask, _mm512_castpd512_pd256(answer));
                if (o1) _mm256_mask_storeu_pd(o1 + f * na + first, mask, _mm512_extractf64x4_pd(answer, 1));
              }
            }
          }
        }
      }
    };
    Shard(workers->num_threads, workers->workers, nchunks,
          std::max<int64_t>(1, (nblocks / nchunks) * block * (np + nt + nf) *
                              (mode == 0 ? 20 : mode == 1 ? 50 : 110)), work);
    if constexpr (compute_weights)
      for (int64_t chunk = 0; chunk < nchunks; ++chunk)
        for (int64_t k = 0; k < nw; ++k) ow[k] += partial_weights[chunk * nw + k];
  };
  if (ow) {
    if (zero_weight_direction) dispatch(std::true_type{}, std::true_type{});
    else dispatch(std::true_type{}, std::false_type{});
  } else {
    if (zero_weight_direction) dispatch(std::false_type{}, std::true_type{});
    else dispatch(std::false_type{}, std::false_type{});
  }
}
