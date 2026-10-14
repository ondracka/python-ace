// All-element primitive moments and rank-one sums in one stable reduction.
// Float64 analytic first/second derivatives, with no fitted coefficients here.
#include <algorithm>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

#define ALL_MOMENT_INPUTS \
    .Input("radial: double").Input("yr: double").Input("yi: double") \
    .Input("segments: int32").Input("nsegments: int32")

REGISTER_OP("WbcAllMoments") ALL_MOMENT_INPUTS
    .Output("ar: double").Output("ai: double").Output("g: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle r, y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &r));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y));
      auto moments = c->MakeShape({c->UnknownDim(), c->Dim(r, 1), c->Dim(y, 1)});
      c->set_output(0, moments);
      c->set_output(1, moments);
      c->set_output(2, c->Matrix(c->UnknownDim(), c->Dim(r, 1)));
      return absl::OkStatus();
    });

REGISTER_OP("WbcAllMomentsGrad") ALL_MOMENT_INPUTS
    .Input("sr: double").Input("si: double").Input("sg: double")
    .Output("dr: double").Output("dyr: double").Output("dyi: double")
    .SetShapeFn([](InferenceContext* c) {
      for (int k = 0; k < 3; ++k) c->set_output(k, c->input(k));
      return absl::OkStatus();
    });

REGISTER_OP("WbcAllMomentsGradGrad") ALL_MOMENT_INPUTS
    .Input("sr: double").Input("si: double").Input("sg: double")
    .Input("hr: double").Input("hyr: double").Input("hyi: double")
    .Output("dr: double").Output("dyr: double").Output("dyi: double")
    .Output("dsr: double").Output("dsi: double").Output("dsg: double")
    .SetShapeFn([](InferenceContext* c) {
      for (int k = 0; k < 3; ++k) {
        c->set_output(k, c->input(k));
        c->set_output(k + 3, c->input(k + 5));
      }
      return absl::OkStatus();
    });

template <int mode>
class WbcAllMomentsKernel : public OpKernel {
 public:
  explicit WbcAllMomentsKernel(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) override {
    const Tensor& radial = c->input(0);
    const Tensor& yr_tensor = c->input(1);
    OP_REQUIRES(c, radial.dims() == 2 && yr_tensor.dims() == 2 &&
                c->input(2).shape() == yr_tensor.shape() &&
                c->input(3).dims() == 1 && c->input(4).dims() == 0,
                errors::InvalidArgument("Invalid all-moment input shapes"));
    const int64_t np = radial.dim_size(0), nr = radial.dim_size(1);
    const int64_t nm = yr_tensor.dim_size(1), ns = c->input(4).scalar<int32>()();
    OP_REQUIRES(c, yr_tensor.dim_size(0) == np && c->input(3).NumElements() == np &&
                nr >= 1 && nm >= 1 && ns >= 0,
                errors::InvalidArgument("Invalid all-moment dimensions"));
    const TensorShape ashape({ns, nr, nm}), gshape({ns, nr});
    if constexpr (mode >= 1) {
      OP_REQUIRES(c, c->input(5).shape() == ashape && c->input(6).shape() == ashape &&
                  c->input(7).shape() == gshape,
                  errors::InvalidArgument("All-moment derivative seed shapes differ"));
    }
    if constexpr (mode == 2) {
      for (int k = 0; k < 3; ++k) {
        OP_REQUIRES(c, c->input(k + 8).shape() == c->input(k).shape(),
                    errors::InvalidArgument("All-moment derivative direction shapes differ"));
      }
    }

    // Stable CSR grouping preserves the input order within each atom/type
    // segment, even when the global neighbour list is interleaved.
    const int32* segments = c->input(3).flat<int32>().data();
    std::vector<int64_t> rows(ns + 1, 0), pairs(np);
    for (int64_t p = 0; p < np; ++p) {
      OP_REQUIRES(c, segments[p] >= 0 && segments[p] < ns,
                  errors::InvalidArgument("All-moment segment index out of range"));
      ++rows[segments[p] + 1];
    }
    for (int64_t s = 0; s < ns; ++s) rows[s + 1] += rows[s];
    std::vector<int64_t> cursor(rows.begin(), rows.end() - 1);
    for (int64_t p = 0; p < np; ++p) pairs[cursor[segments[p]]++] = p;

    const bool seed_only = mode == 2 && !c->output_required(0) &&
        !c->output_required(1) && !c->output_required(2);
    double* out[6] = {};
    const int noutputs = mode == 2 ? 6 : 3;
    for (int k = 0; k < noutputs; ++k) {
      if (seed_only && k < 3) continue;
      Tensor* tensor;
      const TensorShape shape = mode == 0 ? (k == 2 ? gshape : ashape) :
          (k < 3 ? c->input(k).shape() : (k == 5 ? gshape : ashape));
      OP_REQUIRES_OK(c, c->allocate_output(k, shape, &tensor));
      out[k] = tensor->flat<double>().data();
    }

    const double* r = radial.flat<double>().data();
    const double* yr = yr_tensor.flat<double>().data();
    const double* yi = c->input(2).flat<double>().data();
    const double* sr = mode >= 1 ? c->input(5).flat<double>().data() : nullptr;
    const double* si = mode >= 1 ? c->input(6).flat<double>().data() : nullptr;
    const double* sg = mode >= 1 ? c->input(7).flat<double>().data() : nullptr;
    const double* hr = mode == 2 ? c->input(8).flat<double>().data() : nullptr;
    const double* hyr = mode == 2 ? c->input(9).flat<double>().data() : nullptr;
    const double* hyi = mode == 2 ? c->input(10).flat<double>().data() : nullptr;

    auto dispatch = [&](auto seed_only_tag) {
      constexpr bool seeds_only = decltype(seed_only_tag)::value;
      auto work = [&](int64_t begin, int64_t end) {
        for (int64_t s = begin; s < end; ++s) {
          const int64_t abase = s * nr * nm, gbase = s * nr;
          if constexpr (mode == 0 || mode == 2) {
            constexpr int offset = mode == 0 ? 0 : 3;
            std::fill(out[offset] + abase, out[offset] + abase + nr * nm, 0.);
            std::fill(out[offset + 1] + abase, out[offset + 1] + abase + nr * nm, 0.);
            std::fill(out[offset + 2] + gbase, out[offset + 2] + gbase + nr, 0.);
          }
          for (int64_t i = rows[s]; i < rows[s + 1]; ++i) {
            const int64_t p = pairs[i], rbase = p * nr, ybase = p * nm;
            if constexpr (mode >= 1 && !seeds_only) {
              std::fill(out[1] + ybase, out[1] + ybase + nm, 0.);
              std::fill(out[2] + ybase, out[2] + ybase + nm, 0.);
            }
            for (int64_t n = 0; n < nr; ++n) {
              const int64_t ri = rbase + n, gi = gbase + n;
              const double rv = r[ri];
              double dr = 0.;
              if constexpr (mode == 0) out[2][gi] += rv;
              if constexpr (mode == 2) out[5][gi] += hr[ri];
              for (int64_t m = 0; m < nm; ++m) {
                const int64_t a = abase + n * nm + m, y = ybase + m;
                if constexpr (mode == 0) {
                  out[0][a] += rv * yr[y];
                  out[1][a] += rv * yi[y];
                } else if constexpr (mode == 1) {
                  dr += sr[a] * yr[y] + si[a] * yi[y];
                  out[1][y] += sr[a] * rv;
                  out[2][y] += si[a] * rv;
                } else {
                  if constexpr (!seeds_only) {
                    dr += sr[a] * hyr[y] + si[a] * hyi[y];
                    out[1][y] += sr[a] * hr[ri];
                    out[2][y] += si[a] * hr[ri];
                  }
                  out[3][a] += hr[ri] * yr[y] + rv * hyr[y];
                  out[4][a] += hr[ri] * yi[y] + rv * hyi[y];
                }
              }
              if constexpr (mode == 1) out[0][ri] = dr + sg[gi];
              if constexpr (mode == 2 && !seeds_only) out[0][ri] = dr;
            }
          }
        }
      };
      const auto* workers = c->device()->tensorflow_cpu_worker_threads();
      Shard(workers->num_threads, workers->workers, ns,
            std::max<int64_t>(1, np / std::max<int64_t>(1, ns) * nr * nm *
                                (mode == 0 ? 5 : 15)), work);
    };
    if (seed_only) dispatch(std::true_type{});
    else dispatch(std::false_type{});
  }
};

REGISTER_KERNEL_BUILDER(Name("WbcAllMoments").Device(DEVICE_CPU), WbcAllMomentsKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcAllMomentsGrad").Device(DEVICE_CPU), WbcAllMomentsKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcAllMomentsGradGrad").Device(DEVICE_CPU), WbcAllMomentsKernel<2>);
