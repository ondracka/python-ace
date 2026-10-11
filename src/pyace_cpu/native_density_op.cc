// Ordered complex product DAG contracted directly into trainable ACE densities.
// Float64, analytic first/second derivatives, bounded atom-block workspaces.
#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

#define DENSITY_INPUTS \
    .Input("ar: double").Input("ai: double") \
    .Input("left: int32").Input("right: int32") \
    .Input("terms: int32").Input("weights: double")

REGISTER_OP("WbcDensity") DENSITY_INPUTS
    .Output("rho: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle a, w;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 2, &w));
      c->set_output(0, c->Matrix(c->Dim(w, 1), c->Dim(a, 1)));
      return absl::OkStatus();
    });

REGISTER_OP("WbcDensityGrad") DENSITY_INPUTS
    .Input("seed: double")
    .Output("dar: double").Output("dai: double").Output("dweights: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(5));
      return absl::OkStatus();
    });

REGISTER_OP("WbcDensityGradGrad") DENSITY_INPUTS
    .Input("seed: double").Input("hr: double").Input("hi: double").Input("hw: double")
    .Output("dar: double").Output("dai: double").Output("dweights: double")
    .Output("dseed: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(5));
      c->set_output(3, c->input(6));
      return absl::OkStatus();
    });

template <int mode>
class WbcDensityKernel : public OpKernel {
 public:
  explicit WbcDensityKernel(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& a = c->input(0);
    const Tensor& w = c->input(5);
    OP_REQUIRES(c, a.dims() == 2 && a.shape() == c->input(1).shape() && w.dims() == 2,
                errors::InvalidArgument("Density inputs must be compatible matrices"));
    for (int i = 2; i <= 4; ++i) {
      OP_REQUIRES(c, c->input(i).dims() == 1,
                  errors::InvalidArgument("Density plan inputs must be vectors"));
    }
    const int64_t nf = a.dim_size(0), na = a.dim_size(1);
    const int64_t np = c->input(2).NumElements(), nn = nf + np;
    const int64_t nt = c->input(4).NumElements(), nd = w.dim_size(1), nw = nt * nd;
    OP_REQUIRES(c, c->input(3).NumElements() == np && w.dim_size(0) == nt,
                errors::InvalidArgument("Inconsistent density plan lengths"));
    const int32* left = c->input(2).flat<int32>().data();
    const int32* right = c->input(3).flat<int32>().data();
    const int32* terms = c->input(4).flat<int32>().data();
    for (int64_t k = 0; k < np; ++k) {
      OP_REQUIRES(c, left[k] >= 0 && left[k] < nf + k && right[k] >= 0 && right[k] < nf + k,
                  errors::InvalidArgument("Density products must form a topological DAG"));
    }
    for (int64_t t = 0; t < nt; ++t) {
      OP_REQUIRES(c, terms[t] >= 0 && terms[t] < nn && (t == 0 || terms[t - 1] < terms[t]),
                  errors::InvalidArgument("Density term nodes must be sorted and unique"));
    }
    const double* ar = a.flat<double>().data();
    const double* ai = c->input(1).flat<double>().data();
    const double* weights = w.flat<double>().data();
    const double *seed = nullptr, *hr = nullptr, *hi = nullptr, *hw = nullptr;
    if constexpr (mode >= 1) {
      OP_REQUIRES(c, c->input(6).shape() == TensorShape({nd, na}),
                  errors::InvalidArgument("Density derivative seed has wrong shape"));
      seed = c->input(6).flat<double>().data();
    }
    if constexpr (mode == 2) {
      OP_REQUIRES(c, c->input(7).shape() == a.shape() && c->input(8).shape() == a.shape() &&
                  c->input(9).shape() == w.shape(),
                  errors::InvalidArgument("Density derivative directions have wrong shapes"));
      hr = c->input(7).flat<double>().data();
      hi = c->input(8).flat<double>().data();
      hw = c->input(9).flat<double>().data();
    }
    Tensor *out0, *out1 = nullptr, *outw = nullptr, *outs = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, mode == 0 ? TensorShape({nd, na}) : a.shape(), &out0));
    if constexpr (mode >= 1) {
      OP_REQUIRES_OK(c, c->allocate_output(1, a.shape(), &out1));
      OP_REQUIRES_OK(c, c->allocate_output(2, w.shape(), &outw));
      outw->flat<double>().setZero();
    }
    if constexpr (mode == 2) {
      OP_REQUIRES_OK(c, c->allocate_output(3, TensorShape({nd, na}), &outs));
    }
    if (na == 0) return;
    double* o0 = out0->flat<double>().data();
    double* o1 = out1 ? out1->flat<double>().data() : nullptr;
    double* ow = outw ? outw->flat<double>().data() : nullptr;
    double* os = outs ? outs->flat<double>().data() : nullptr;
    constexpr int block = 4;
    const int64_t nblocks = (na + block - 1) / block;
    const auto* workers = c->device()->tensorflow_cpu_worker_threads();
    const int64_t nchunks = std::min<int64_t>(nblocks, std::max(1, workers->num_threads));
    // Each fixed chunk has its own coefficient reduction. Merge in chunk order.
    std::vector<double> partial_weights(mode >= 1 ? nchunks * nw : 0, 0.);
    auto work = [&](int64_t begin, int64_t end) {
      std::vector<double> pr(nn * block), pi(nn * block), gr, gi, dr, di, dgr, dgi;
      std::vector<double> rho(nd * block), s(nd * block);
      if constexpr (mode >= 1) { gr.resize(nn * block); gi.resize(nn * block); }
      if constexpr (mode == 2) {
        dr.resize(nn * block); di.resize(nn * block);
        dgr.resize(nn * block); dgi.resize(nn * block);
      }
      for (int64_t chunk = begin; chunk < end; ++chunk) {
        double* gw = mode >= 1 ? partial_weights.data() + chunk * nw : nullptr;
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
          for (int64_t k = 0; k < np; ++k) {
            const int64_t node = (nf + k) * block, lhs = left[k] * block, rhs = right[k] * block;
            #pragma GCC ivdep
            for (int lane = 0; lane < block; ++lane) {
              const int64_t n = node + lane, l = lhs + lane, r = rhs + lane;
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
            for (int64_t d = 0; d < nd; ++d) {
              for (int lane = 0; lane < block; ++lane)
                s[d * block + lane] = lane < valid ? seed[d * na + first + lane] : 0.;
            }
          }
          if constexpr (mode == 2) {
            std::fill(dgr.begin(), dgr.end(), 0.);
            std::fill(dgi.begin(), dgi.end(), 0.);
          }
          for (int64_t t = 0; t < nt; ++t) {
            const int64_t node = terms[t] * block;
            for (int64_t d = 0; d < nd; ++d) {
              const int64_t wd = t * nd + d, ds = d * block;
              const double coefficient = weights[wd];
              if constexpr (mode == 0) {
                #pragma GCC ivdep
                for (int lane = 0; lane < block; ++lane)
                  rho[ds + lane] += coefficient * pr[node + lane];
              } else {
                #pragma GCC ivdep
                for (int lane = 0; lane < block; ++lane) {
                  gr[node + lane] += coefficient * s[ds + lane];
                  if constexpr (mode == 2) {
                    dgr[node + lane] += hw[wd] * s[ds + lane];
                    rho[ds + lane] += coefficient * dr[node + lane] + hw[wd] * pr[node + lane];
                  }
                }
                for (int lane = 0; lane < valid; ++lane) {
                  if constexpr (mode == 1) gw[wd] += pr[node + lane] * s[ds + lane];
                  else gw[wd] += dr[node + lane] * s[ds + lane];
                }
              }
            }
          }
          if constexpr (mode == 0 || mode == 2) {
            double* target = mode == 0 ? o0 : os;
            for (int64_t d = 0; d < nd; ++d) {
              for (int lane = 0; lane < valid; ++lane)
                target[d * na + first + lane] = rho[d * block + lane];
            }
          }
          if constexpr (mode >= 1) {
            for (int64_t k = np; k-- > 0;) {
              const int64_t node = (nf + k) * block, lhs = left[k] * block, rhs = right[k] * block;
              #pragma GCC ivdep
              for (int lane = 0; lane < block; ++lane) {
                const int64_t n = node + lane, l = lhs + lane, r = rhs + lane;
                if constexpr (mode == 2) {
                  dgr[l] += (dgr[n] * pr[r] + dgi[n] * pi[r]) + (gr[n] * dr[r] + gi[n] * di[r]);
                  dgi[l] += (dgi[n] * pr[r] - dgr[n] * pi[r]) + (gi[n] * dr[r] - gr[n] * di[r]);
                  dgr[r] += (dgr[n] * pr[l] + dgi[n] * pi[l]) + (gr[n] * dr[l] + gi[n] * di[l]);
                  dgi[r] += (dgi[n] * pr[l] - dgr[n] * pi[l]) + (gi[n] * dr[l] - gr[n] * di[l]);
                }
                gr[l] += gr[n] * pr[r] + gi[n] * pi[r];
                gi[l] += gi[n] * pr[r] - gr[n] * pi[r];
                gr[r] += gr[n] * pr[l] + gi[n] * pi[l];
                gi[r] += gi[n] * pr[l] - gr[n] * pi[l];
              }
            }
            for (int64_t f = 0; f < nf; ++f) {
              for (int lane = 0; lane < valid; ++lane) {
                o0[f * na + first + lane] = mode == 1 ? gr[f * block + lane] : dgr[f * block + lane];
                o1[f * na + first + lane] = mode == 1 ? gi[f * block + lane] : dgi[f * block + lane];
              }
            }
          }
        }
      }
    };
    Shard(workers->num_threads, workers->workers, nchunks,
          std::max<int64_t>(1, (nblocks / nchunks) * block * (np + nt + nf) *
                              (mode == 0 ? 20 : mode == 1 ? 50 : 110)), work);
    if constexpr (mode >= 1) {
      for (int64_t chunk = 0; chunk < nchunks; ++chunk) {
        for (int64_t k = 0; k < nw; ++k) ow[k] += partial_weights[chunk * nw + k];
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("WbcDensity").Device(DEVICE_CPU), WbcDensityKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcDensityGrad").Device(DEVICE_CPU), WbcDensityKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcDensityGradGrad").Device(DEVICE_CPU), WbcDensityKernel<2>);
