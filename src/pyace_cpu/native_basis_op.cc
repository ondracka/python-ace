// Ordered ACE product DAG and exact first/second derivatives in float64.
// Plans contain fixed basis indices and Clebsch-Gordan weights, not fit weights.
#include <algorithm>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

#define BASIS_INPUTS \
    .Input("ar: double").Input("ai: double") \
    .Input("left: int32").Input("right: int32") \
    .Input("row_ptr: int32").Input("terms: int32").Input("weights: double") \
    .Attr("block_size: int = 1")

REGISTER_OP("WbcBasis") BASIS_INPUTS
    .Output("basis: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle a;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));
      c->set_output(0, c->Matrix(c->UnknownDim(), c->Dim(a, 1)));
      return absl::OkStatus();
    });

REGISTER_OP("WbcBasisGrad") BASIS_INPUTS
    .Input("seed: double").Output("dar: double").Output("dai: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return absl::OkStatus();
    });

REGISTER_OP("WbcBasisGradGrad") BASIS_INPUTS
    .Input("seed: double").Input("hr: double").Input("hi: double")
    .Output("dar: double").Output("dai: double").Output("dseed: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(7));
      return absl::OkStatus();
    });

template <int mode>
class WbcBasisKernel : public OpKernel {
 public:
  explicit WbcBasisKernel(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("block_size", &block_size_));
    OP_REQUIRES(context, block_size_ == 1 || block_size_ == 4 || block_size_ == 8,
                errors::InvalidArgument("Supported ACE atom block sizes are 1, 4 and 8"));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& ar_tensor = c->input(0);
    OP_REQUIRES(c, ar_tensor.dims() == 2 && ar_tensor.shape() == c->input(1).shape(),
                errors::InvalidArgument("ACE real/imaginary inputs must be equal matrices"));
    for (int k = 2; k < 7; ++k) {
      OP_REQUIRES(c, c->input(k).dims() == 1,
                  errors::InvalidArgument("ACE plan inputs must be vectors"));
    }
    const int64_t nf = ar_tensor.dim_size(0), na = ar_tensor.dim_size(1);
    const int64_t np = c->input(2).NumElements(), nn = nf + np;
    const int64_t nb = c->input(4).NumElements() - 1;
    const int64_t nt = c->input(5).NumElements();
    OP_REQUIRES(c, nb >= 0 && c->input(3).NumElements() == np &&
                c->input(6).NumElements() == nt,
                errors::InvalidArgument("Inconsistent ACE plan lengths"));
    const int32* left = c->input(2).flat<int32>().data();
    const int32* right = c->input(3).flat<int32>().data();
    const int32* row = c->input(4).flat<int32>().data();
    const int32* terms = c->input(5).flat<int32>().data();
    const double* weights = c->input(6).flat<double>().data();
    OP_REQUIRES(c, row[0] == 0 && row[nb] == nt,
                errors::InvalidArgument("Invalid ACE CG row boundaries"));
    for (int64_t k = 0; k < np; ++k) {
      OP_REQUIRES(c, left[k] >= 0 && left[k] < nf + k && right[k] >= 0 && right[k] < nf + k,
                  errors::InvalidArgument("ACE products must form a topological DAG"));
    }
    for (int64_t k = 0; k < nb; ++k) {
      OP_REQUIRES(c, row[k] >= 0 && row[k] <= row[k + 1] && row[k + 1] <= nt,
                  errors::InvalidArgument("ACE CG rows must be sorted"));
    }
    for (int64_t k = 0; k < nt; ++k) {
      OP_REQUIRES(c, terms[k] >= 0 && terms[k] < nn,
                  errors::InvalidArgument("ACE CG product index out of range"));
    }
    const double* ar = ar_tensor.flat<double>().data();
    const double* ai = c->input(1).flat<double>().data();
    const double* seed = nullptr;
    const double* hr = nullptr;
    const double* hi = nullptr;
    if constexpr (mode >= 1) {
      OP_REQUIRES(c, c->input(7).shape() == TensorShape({nb, na}),
                  errors::InvalidArgument("ACE derivative seed has wrong shape"));
      seed = c->input(7).flat<double>().data();
    }
    if constexpr (mode == 2) {
      OP_REQUIRES(c, c->input(8).shape() == ar_tensor.shape() &&
                  c->input(9).shape() == ar_tensor.shape(),
                  errors::InvalidArgument("ACE derivative directions have wrong shape"));
      hr = c->input(8).flat<double>().data();
      hi = c->input(9).flat<double>().data();
    }
    Tensor* out0;
    Tensor* out1 = nullptr;
    Tensor* out2 = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, mode == 0 ? TensorShape({nb, na}) : ar_tensor.shape(), &out0));
    if constexpr (mode >= 1) {
      OP_REQUIRES_OK(c, c->allocate_output(1, ar_tensor.shape(), &out1));
    }
    if constexpr (mode == 2) {
      OP_REQUIRES_OK(c, c->allocate_output(2, TensorShape({nb, na}), &out2));
    }
    double* o0 = out0->flat<double>().data();
    double* o1 = out1 ? out1->flat<double>().data() : nullptr;
    double* o2 = out2 ? out2->flat<double>().data() : nullptr;
    if (na == 0) return;

    auto dispatch = [&](auto block_tag) {
    constexpr int block = decltype(block_tag)::value;
    auto work = [=](int64_t begin, int64_t end) {
      // Reuse small per-thread workspaces across atoms, never dataset-sized caches.
      std::vector<double> pr(nn * block), pi(nn * block), gr, gi, dr, di, dgr, dgi;
      if constexpr (mode >= 1) { gr.resize(nn * block); gi.resize(nn * block); }
      if constexpr (mode == 2) {
        dr.resize(nn * block); di.resize(nn * block);
        dgr.resize(nn * block); dgi.resize(nn * block);
      }
      for (int64_t atom_block = begin; atom_block < end; ++atom_block) {
        const int64_t first_atom = atom_block * block;
        const int valid = std::min<int64_t>(block, na - first_atom);
        for (int64_t f = 0; f < nf; ++f) {
          for (int lane = 0; lane < block; ++lane) {
            const int64_t source = f * na + first_atom + lane, dest = f * block + lane;
            pr[dest] = lane < valid ? ar[source] : 0.;
            pi[dest] = lane < valid ? ai[source] : 0.;
            if constexpr (mode == 2) {
              dr[dest] = lane < valid ? hr[source] : 0.;
              di[dest] = lane < valid ? hi[source] : 0.;
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
        if constexpr (mode == 0) {
          for (int64_t b = 0; b < nb; ++b) {
            double value[block] = {};
            for (int64_t t = row[b]; t < row[b + 1]; ++t) {
              #pragma GCC ivdep
              for (int lane = 0; lane < block; ++lane)
                value[lane] += weights[t] * pr[terms[t] * block + lane];
            }
            for (int lane = 0; lane < valid; ++lane) o0[b * na + first_atom + lane] = value[lane];
          }
        } else {
          std::fill(gr.begin(), gr.end(), 0.);
          std::fill(gi.begin(), gi.end(), 0.);
          if constexpr (mode == 2) {
            std::fill(dgr.begin(), dgr.end(), 0.);
            std::fill(dgi.begin(), dgi.end(), 0.);
          }
          for (int64_t b = 0; b < nb; ++b) {
            double directional[block] = {}, seed_value[block] = {};
            for (int lane = 0; lane < valid; ++lane) seed_value[lane] = seed[b * na + first_atom + lane];
            for (int64_t t = row[b]; t < row[b + 1]; ++t) {
              #pragma GCC ivdep
              for (int lane = 0; lane < block; ++lane) {
                gr[terms[t] * block + lane] += weights[t] * seed_value[lane];
                if constexpr (mode == 2) directional[lane] += weights[t] * dr[terms[t] * block + lane];
              }
            }
            if constexpr (mode == 2) {
              for (int lane = 0; lane < valid; ++lane) o2[b * na + first_atom + lane] = directional[lane];
            }
          }
          for (int64_t k = np; k-- > 0;) {
            const int64_t node = (nf + k) * block, lhs = left[k] * block, rhs = right[k] * block;
            // Different lanes address disjoint atoms, even when lhs == rhs.
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
              o0[f * na + first_atom + lane] = mode == 1 ? gr[f * block + lane] : dgr[f * block + lane];
              o1[f * na + first_atom + lane] = mode == 1 ? gi[f * block + lane] : dgi[f * block + lane];
            }
          }
        }
      }
    };
    const auto* workers = c->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, (na + block - 1) / block,
          std::max<int64_t>(1, block * (np + nt + nf) * (mode == 0 ? 15 : mode == 1 ? 45 : 100)), work);
    };
    if (block_size_ == 4) dispatch(std::integral_constant<int, 4>{});
    else if (block_size_ == 8) dispatch(std::integral_constant<int, 8>{});
    else dispatch(std::integral_constant<int, 1>{});
  }
 private:
  int block_size_;
};

REGISTER_KERNEL_BUILDER(Name("WbcBasis").Device(DEVICE_CPU), WbcBasisKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcBasisGrad").Device(DEVICE_CPU), WbcBasisKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcBasisGradGrad").Device(DEVICE_CPU), WbcBasisKernel<2>);
