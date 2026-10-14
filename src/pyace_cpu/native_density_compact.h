// Terminal product nodes are consumed directly, without full-node work arrays.
// The product DAG, contraction order and reverse accumulation order are retained.
#pragma once

#include <array>

template <int mode>
void CompactDensity(
    const int64_t nf, const int64_t na, const int64_t np,
    const int64_t nt, const int64_t nd, const int32* left,
    const int32* right, const int32* terms,
    const std::vector<unsigned char>& is_parent,
    const double* ar, const double* ai, const double* weights,
    const double* seed, const double* hr, const double* hi, const double* hw,
    double* o0, double* o1, double* ow, double* os,
    const bool zero_weight_direction, OpKernelContext* context) {
  constexpr int block = WBC_DENSITY_BLOCK;
  const int64_t nn = nf + np, nw = nt * nd;
  const bool reverse = mode >= 1 && (o0 || o1);
  const int64_t nblocks = (na + block - 1) / block;
  std::vector<int64_t> slots(nn, -1), term_ids(nn, -1), lhs(np), rhs(np);
  int64_t ns = nf;
  for (int64_t f = 0; f < nf; ++f) slots[f] = f;
  for (int64_t k = 0; k < np; ++k) {
    if (is_parent[nf + k]) slots[nf + k] = ns++;
    lhs[k] = slots[left[k]] * block;
    rhs[k] = slots[right[k]] * block;
  }
  for (int64_t t = 0; t < nt; ++t) term_ids[terms[t]] = t;
  const auto* workers = context->device()->tensorflow_cpu_worker_threads();
  const int64_t nchunks = std::min<int64_t>(nblocks, std::max(1, workers->num_threads));

  auto dispatch = [&](auto weight_tag, auto zero_tag) {
    constexpr bool compute_weights = decltype(weight_tag)::value;
    constexpr bool zero_hw = decltype(zero_tag)::value;
    std::vector<double> partial_weights(compute_weights ? nchunks * nw : 0, 0.);
    auto work = [&](int64_t begin, int64_t end) {
      std::vector<double> pr(ns * block), pi(ns * block), dr, di, gr, gi, dgr, dgi;
      std::vector<double> rho(nd * block), s(nd * block);
      if (reverse) { gr.resize(ns * block); gi.resize(ns * block); }
      if constexpr (mode == 2) {
        dr.resize(ns * block); di.resize(ns * block);
        if (reverse) { dgr.resize(ns * block); dgi.resize(ns * block); }
      }
      for (int64_t chunk = begin; chunk < end; ++chunk) {
        double* gw = compute_weights ? partial_weights.data() + chunk * nw : nullptr;
        const int64_t first_block = nblocks * chunk / nchunks;
        const int64_t last_block = nblocks * (chunk + 1) / nchunks;
        for (int64_t atom_block = first_block; atom_block < last_block; ++atom_block) {
          const int64_t first = atom_block * block;
          const int valid = std::min<int64_t>(block, na - first);
          for (int64_t f = 0; f < nf; ++f) {
            for (int lane = 0; lane < block; ++lane) {
              const int64_t src = f * na + first + lane, dst = f * block + lane;
              pr[dst] = lane < valid ? ar[src] : 0.;
              pi[dst] = lane < valid ? ai[src] : 0.;
              if constexpr (mode == 2) {
                dr[dst] = lane < valid ? hr[src] : 0.;
                di[dst] = lane < valid ? hi[src] : 0.;
              }
            }
          }
          // Only intermediates used by later products need persistent storage.
          for (int64_t k = 0; k < np; ++k) {
            if (slots[nf + k] < 0) continue;
            const int64_t node = slots[nf + k] * block;
            #pragma GCC ivdep
            for (int lane = 0; lane < block; ++lane) {
              const int64_t n = node + lane, l = lhs[k] + lane, r = rhs[k] + lane;
              pr[n] = pr[l] * pr[r] - pi[l] * pi[r];
              pi[n] = pi[r] * pr[l] + pi[l] * pr[r];
              if constexpr (mode == 2) {
                dr[n] = (dr[l] * pr[r] - di[l] * pi[r]) + (pr[l] * dr[r] - pi[l] * di[r]);
                di[n] = (pi[r] * dr[l] + di[l] * pr[r]) + (di[r] * pr[l] + pi[l] * dr[r]);
              }
            }
          }
          std::fill(rho.begin(), rho.end(), 0.);
          if constexpr (mode >= 1) {
            std::fill(gr.begin(), gr.end(), 0.);
            std::fill(gi.begin(), gi.end(), 0.);
            for (int64_t d = 0; d < nd; ++d)
              for (int lane = 0; lane < block; ++lane)
                s[d * block + lane] = lane < valid ? seed[d * na + first + lane] : 0.;
          }
          if constexpr (mode == 2) {
            std::fill(dgr.begin(), dgr.end(), 0.);
            std::fill(dgi.begin(), dgi.end(), 0.);
          }
          for (int64_t t = 0; t < nt; ++t) {
            const int64_t node = slots[terms[t]];
            std::array<double, block> value{}, tangent{};
            const bool need_value = mode == 0 || (mode == 1 && compute_weights) ||
                                    (mode == 2 && !zero_hw && os);
            const bool need_tangent = mode == 2 && (compute_weights || os);
            for (int lane = 0; lane < block; ++lane) {
              if (node >= 0) {
                if (need_value) value[lane] = pr[node * block + lane];
                if (need_tangent) tangent[lane] = dr[node * block + lane];
              } else {
                const int64_t k = terms[t] - nf, l = lhs[k] + lane, r = rhs[k] + lane;
                if (need_value) value[lane] = pr[l] * pr[r] - pi[l] * pi[r];
                if (need_tangent)
                  tangent[lane] = (dr[l] * pr[r] - di[l] * pi[r]) + (pr[l] * dr[r] - pi[l] * di[r]);
              }
            }
            for (int64_t d = 0; d < nd; ++d) {
              const int64_t wd = t * nd + d, ds = d * block;
              const double coefficient = weights[wd];
              if constexpr (mode == 0) {
                #pragma GCC ivdep
                for (int lane = 0; lane < block; ++lane)
                  rho[ds + lane] += coefficient * value[lane];
              } else {
                #pragma GCC ivdep
                for (int lane = 0; lane < block; ++lane) {
                  if (reverse && node >= 0) gr[node * block + lane] += coefficient * s[ds + lane];
                  if constexpr (mode == 2) {
                    if constexpr (!zero_hw) {
                      if (reverse && node >= 0) dgr[node * block + lane] += hw[wd] * s[ds + lane];
                    }
                    if (os) {
                      if constexpr (zero_hw) rho[ds + lane] += coefficient * tangent[lane];
                      else rho[ds + lane] += coefficient * tangent[lane] + hw[wd] * value[lane];
                    }
                  }
                }
                if constexpr (compute_weights) {
                  for (int lane = 0; lane < valid; ++lane) {
                    if constexpr (mode == 1) gw[wd] += value[lane] * s[ds + lane];
                    else gw[wd] += tangent[lane] * s[ds + lane];
                  }
                }
              }
            }
          }
          if constexpr (mode == 0 || mode == 2) {
            double* target = mode == 0 ? o0 : os;
            if (target)
              for (int64_t d = 0; d < nd; ++d)
                for (int lane = 0; lane < valid; ++lane)
                  target[d * na + first + lane] = rho[d * block + lane];
          }
          if constexpr (mode >= 1) {
            if (reverse) {
              for (int64_t k = np; k-- > 0;) {
                const int64_t node = slots[nf + k];
                std::array<double, block> terminal_gr{}, terminal_dgr{};
                const double *qr, *qi, *qdr = nullptr, *qdi = nullptr;
                if (node >= 0) {
                  qr = gr.data() + node * block; qi = gi.data() + node * block;
                  if constexpr (mode == 2) {
                    qdr = dgr.data() + node * block; qdi = dgi.data() + node * block;
                  }
                } else {
                  const int64_t t = term_ids[nf + k];
                  if (t < 0) continue;
                  for (int64_t d = 0; d < nd; ++d) {
                    #pragma GCC ivdep
                    for (int lane = 0; lane < block; ++lane) {
                      terminal_gr[lane] += weights[t * nd + d] * s[d * block + lane];
                      if constexpr (mode == 2 && !zero_hw)
                        terminal_dgr[lane] += hw[t * nd + d] * s[d * block + lane];
                    }
                  }
                  // Terminal densities are real, so their imaginary reverse
                  // seeds are structurally zero. Avoid generic complex
                  // arithmetic here, including the identically zero tangent
                  // seed when no coefficient direction is requested.
                  #pragma GCC ivdep
                  for (int lane = 0; lane < block; ++lane) {
                    const int64_t l = lhs[k] + lane, r = rhs[k] + lane;
                    const double q = terminal_gr[lane];
                    if constexpr (mode == 2) {
                      if constexpr (zero_hw) {
                        dgr[l] += q * dr[r];
                        dgi[l] -= q * di[r];
                        dgr[r] += q * dr[l];
                        dgi[r] -= q * di[l];
                      } else {
                        const double dq = terminal_dgr[lane];
                        dgr[l] += dq * pr[r] + q * dr[r];
                        dgi[l] += -dq * pi[r] - q * di[r];
                        dgr[r] += dq * pr[l] + q * dr[l];
                        dgi[r] += -dq * pi[l] - q * di[l];
                      }
                    }
                    // Sequential updates also handle repeated factors l == r.
                    gr[l] += q * pr[r];
                    gi[l] -= q * pi[r];
                    gr[r] += q * pr[l];
                    gi[r] -= q * pi[l];
                  }
                  continue;
                }
                #pragma GCC ivdep
                for (int lane = 0; lane < block; ++lane) {
                  const int64_t l = lhs[k] + lane, r = rhs[k] + lane;
                  if constexpr (mode == 2) {
                    dgr[l] += (qdr[lane] * pr[r] + qdi[lane] * pi[r]) + (qr[lane] * dr[r] + qi[lane] * di[r]);
                    dgi[l] += (qdi[lane] * pr[r] - qdr[lane] * pi[r]) + (qi[lane] * dr[r] - qr[lane] * di[r]);
                    dgr[r] += (qdr[lane] * pr[l] + qdi[lane] * pi[l]) + (qr[lane] * dr[l] + qi[lane] * di[l]);
                    dgi[r] += (qdi[lane] * pr[l] - qdr[lane] * pi[l]) + (qi[lane] * dr[l] - qr[lane] * di[l]);
                  }
                  gr[l] += qr[lane] * pr[r] + qi[lane] * pi[r];
                  gi[l] += qi[lane] * pr[r] - qr[lane] * pi[r];
                  gr[r] += qr[lane] * pr[l] + qi[lane] * pi[l];
                  gi[r] += qi[lane] * pr[l] - qr[lane] * pi[l];
                }
              }
              for (int64_t f = 0; f < nf; ++f)
                for (int lane = 0; lane < valid; ++lane) {
                  if (o0) o0[f * na + first + lane] = mode == 1 ? gr[f * block + lane] : dgr[f * block + lane];
                  if (o1) o1[f * na + first + lane] = mode == 1 ? gi[f * block + lane] : dgi[f * block + lane];
                }
            }
          }
        }
      }
    };
    Shard(workers->num_threads, workers->workers, nchunks,
          std::max<int64_t>(1, (nblocks / nchunks) * block * (np + nt + nf) *
                              (mode == 0 ? 20 : mode == 1 ? 50 : 110)), work);
    if constexpr (compute_weights) {
      for (int64_t chunk = 0; chunk < nchunks; ++chunk)
        for (int64_t k = 0; k < nw; ++k) ow[k] += partial_weights[chunk * nw + k];
    }
  };
  if (ow) {
    if (zero_weight_direction) dispatch(std::true_type{}, std::true_type{});
    else dispatch(std::true_type{}, std::false_type{});
  } else {
    if (zero_weight_direction) dispatch(std::false_type{}, std::true_type{});
    else dispatch(std::false_type{}, std::false_type{});
  }
}
