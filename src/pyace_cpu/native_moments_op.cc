// Sum radial/angular neighbour moments without pair-by-n-by-m temporaries.
// Exact float64 first and second derivatives, no geometry or coefficient cache.
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;

#define MOMENT_INPUTS \
    .Input("radial: double").Input("yr: double").Input("yi: double") \
    .Input("segments: int32").Attr("nradout: int >= 1") \
    .Attr("l_dependent: bool = true").Attr("prune_unused: bool = false")

REGISTER_OP("WbcMoments") MOMENT_INPUTS
    .Output("ar: double").Output("ai: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle y;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &y));
      int nr;
      TF_RETURN_IF_ERROR(c->GetAttr("nradout", &nr));
      auto out = c->MakeShape({c->UnknownDim(), c->MakeDim(nr), c->Dim(y, 1)});
      c->set_output(0, out);
      c->set_output(1, out);
      return absl::OkStatus();
    });

REGISTER_OP("WbcMomentsGrad") MOMENT_INPUTS
    .Input("sr: double").Input("si: double")
    .Output("dr: double").Output("dyr: double").Output("dyi: double")
    .SetShapeFn([](InferenceContext* c) {
      for (int k = 0; k < 3; ++k) c->set_output(k, c->input(k));
      return absl::OkStatus();
    });

REGISTER_OP("WbcMomentsGradGrad") MOMENT_INPUTS
    .Input("sr: double").Input("si: double")
    .Input("hr: double").Input("hyr: double").Input("hyi: double")
    .Output("dr: double").Output("dyr: double").Output("dyi: double")
    .Output("dsr: double").Output("dsi: double")
    .SetShapeFn([](InferenceContext* c) {
      for (int k = 0; k < 3; ++k) c->set_output(k, c->input(k));
      c->set_output(3, c->input(4));
      c->set_output(4, c->input(5));
      return absl::OkStatus();
    });

template <int mode>
class WbcMomentsKernel : public OpKernel {
 public:
  explicit WbcMomentsKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("nradout", &nout_));
    OP_REQUIRES_OK(c, c->GetAttr("l_dependent", &l_dependent_));
    OP_REQUIRES_OK(c, c->GetAttr("prune_unused", &prune_unused_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& radial = c->input(0);
    const Tensor& yr_tensor = c->input(1);
    OP_REQUIRES(c, radial.dims() == 3 && yr_tensor.dims() == 2 &&
                c->input(2).shape() == yr_tensor.shape() && c->input(3).dims() == 1,
                errors::InvalidArgument("Invalid radial/angular/segment ranks"));
    const int64_t np = radial.dim_size(0), nr = radial.dim_size(1), nl = radial.dim_size(2);
    const int64_t nm = yr_tensor.dim_size(1), nout = nout_;
    const int64_t global_nl = static_cast<int64_t>(std::sqrt(nm));
    const bool ld = l_dependent_;
    OP_REQUIRES(c, yr_tensor.dim_size(0) == np && c->input(3).NumElements() == np &&
                nr <= nout && nl >= 1 && global_nl * global_nl == nm &&
                (ld ? nl <= global_nl : nl == 1),
                errors::InvalidArgument("Invalid moment dimensions"));
    const int32* seg = c->input(3).flat<int32>().data();
    int64_t ns = 0;
    for (int64_t p = 0; p < np; ++p) {
      OP_REQUIRES(c, seg[p] >= 0 && (p == 0 || seg[p] >= seg[p - 1]),
                  errors::InvalidArgument("Moment segment IDs must be nonnegative and sorted"));
      ns = static_cast<int64_t>(seg[p]) + 1;
    }
    const TensorShape ashape({ns, nout, nm});
    if constexpr (mode >= 1) {
      OP_REQUIRES(c, c->input(4).shape() == ashape && c->input(5).shape() == ashape,
                  errors::InvalidArgument("Moment derivative seed shapes differ"));
    }
    if constexpr (mode == 2) {
      for (int k = 0; k < 3; ++k) {
        OP_REQUIRES(c, c->input(k + 6).shape() == c->input(k).shape(),
                    errors::InvalidArgument("Moment derivative direction shapes differ"));
      }
    }
    std::vector<int64_t> row(ns + 1, 0);
    for (int64_t p = 0; p < np; ++p) ++row[seg[p] + 1];
    for (int64_t s = 0; s < ns; ++s) row[s + 1] += row[s];
    double* out[5] = {};
    // During fixed-geometry coefficient fitting only derivatives of the
    // incoming seeds are needed here. Keep the full kernel for all other uses.
    const bool seed_only = mode == 2 && prune_unused_ &&
        !c->output_required(0) && !c->output_required(1) && !c->output_required(2);
    const int outputs = mode == 0 ? 2 : mode == 1 ? 3 : 5;
    for (int k = 0; k < outputs; ++k) {
      if (seed_only && k < 3) continue;
      Tensor* tensor;
      OP_REQUIRES_OK(c, c->allocate_output(k, mode == 0 || k >= 3 ? ashape : c->input(k).shape(), &tensor));
      out[k] = tensor->flat<double>().data();
    }
    if (np == 0) return;
    const double* r = radial.flat<double>().data();
    const double* yr = yr_tensor.flat<double>().data();
    const double* yi = c->input(2).flat<double>().data();
    const double* sr = mode >= 1 ? c->input(4).flat<double>().data() : nullptr;
    const double* si = mode >= 1 ? c->input(5).flat<double>().data() : nullptr;
    const double* hr = mode == 2 ? c->input(6).flat<double>().data() : nullptr;
    const double* hyr = mode == 2 ? c->input(7).flat<double>().data() : nullptr;
    const double* hyi = mode == 2 ? c->input(8).flat<double>().data() : nullptr;
    auto dispatch = [&](auto seed_only_tag) {
    constexpr bool seeds_only = decltype(seed_only_tag)::value;
    auto work = [&](int64_t begin, int64_t end) {
      for (int64_t s = begin; s < end; ++s) {
        const int64_t abase = s * nout * nm;
        if constexpr (mode == 0) {
          std::fill(out[0] + abase, out[0] + abase + nout * nm, 0.);
          std::fill(out[1] + abase, out[1] + abase + nout * nm, 0.);
        }
        if constexpr (mode == 2) {
          std::fill(out[3] + abase, out[3] + abase + nout * nm, 0.);
          std::fill(out[4] + abase, out[4] + abase + nout * nm, 0.);
        }
        for (int64_t p = row[s]; p < row[s + 1]; ++p) {
          const int64_t rbase = p * nr * nl, ybase = p * nm;
          if constexpr (mode >= 1 && !seeds_only) {
            std::fill(out[0] + rbase, out[0] + rbase + nr * nl, 0.);
            std::fill(out[1] + ybase, out[1] + ybase + nm, 0.);
            std::fill(out[2] + ybase, out[2] + ybase + nm, 0.);
          }
          for (int64_t n = 0; n < nr; ++n) {
            for (int64_t l = 0; l < nl; ++l) {
              const int64_t ri = rbase + n * nl + l;
              const int64_t first = ld ? l * l : 0, stop = ld ? (l + 1) * (l + 1) : nm;
              const double rv = r[ri];
              double dr = 0.;
              for (int64_t m = first; m < stop; ++m) {
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
              if constexpr (mode >= 1 && !seeds_only) out[0][ri] = dr;
            }
          }
        }
      }
    };
    const auto* workers = c->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, ns,
          std::max<int64_t>(1, np / std::max<int64_t>(1, ns) * nr * nm * (mode == 0 ? 5 : 15)), work);
    };
    if constexpr (mode == 2) {
      if (seed_only) dispatch(std::true_type{});
      else dispatch(std::false_type{});
    } else dispatch(std::false_type{});
  }
 private:
  int nout_;
  bool l_dependent_;
  bool prune_unused_;
};

REGISTER_KERNEL_BUILDER(Name("WbcMoments").Device(DEVICE_CPU), WbcMomentsKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcMomentsGrad").Device(DEVICE_CPU), WbcMomentsKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcMomentsGradGrad").Device(DEVICE_CPU), WbcMomentsKernel<2>);
