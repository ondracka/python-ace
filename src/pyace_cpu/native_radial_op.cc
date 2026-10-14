// SBessel recurrence and inner/outer cutoffs, float64 analytic derivatives.
// Only atomic distance is differentiated. The plan and cutoffs are fixed model
// settings, never trainable radial mixing coefficients.
#include <algorithm>
#include <cmath>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
#define RADIAL_INPUTS \
    .Input("r: double").Input("settings: double").Input("plan: double") \
    .Attr("trig_recurrence: bool = false")

REGISTER_OP("WbcRadial") RADIAL_INPUTS .Output("g: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), c->Dim(c->input(2), 0)));
      return absl::OkStatus();
    });
REGISTER_OP("WbcRadialWithDerivative") RADIAL_INPUTS
    .Output("g: double").Output("dg: double")
    .SetShapeFn([](InferenceContext* c) {
      auto shape = c->Matrix(c->Dim(c->input(0), 0), c->Dim(c->input(2), 0));
      c->set_output(0, shape);
      c->set_output(1, shape);
      return absl::OkStatus();
    });
REGISTER_OP("WbcRadialGrad") RADIAL_INPUTS .Input("seed: double").Output("dr: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });
REGISTER_OP("WbcRadialGradGrad") RADIAL_INPUTS
    .Input("seed: double").Input("direction: double")
    .Output("dr: double").Output("dseed: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(3));
      return absl::OkStatus();
    });

template <int mode>
class WbcRadialKernel : public OpKernel {
 public:
  explicit WbcRadialKernel(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("trig_recurrence", &trig_recurrence_));
  }
  void Compute(OpKernelContext* c) override {
    if (trig_recurrence_) ComputeImpl<true>(c);
    else ComputeImpl<false>(c);
  }

  template <bool trig_recurrence>
  void ComputeImpl(OpKernelContext* c) {
    const Tensor& distances = c->input(0);
    OP_REQUIRES(c, distances.dims() == 2 && distances.dim_size(1) == 1 &&
                c->input(1).shape() == TensorShape({3}) && c->input(2).dims() == 2 &&
                c->input(2).dim_size(1) == 3 && c->input(2).dim_size(0) >= 1,
                errors::InvalidArgument("Invalid radial input shapes"));
    const int64_t np = distances.dim_size(0), nf = c->input(2).dim_size(0);
    const double* r = distances.flat<double>().data();
    const double* settings = c->input(1).flat<double>().data();
    const double* plan = c->input(2).flat<double>().data();
    const double rc = settings[0], rin = settings[1], delta = settings[2];
    OP_REQUIRES(c, std::isfinite(rc) && rc > 0. && std::isfinite(rin) &&
                std::isfinite(delta) && delta >= 0.,
                errors::InvalidArgument("Invalid fixed radial cutoffs"));
    if constexpr (mode == 1 || mode == 2) {
      OP_REQUIRES(c, c->input(3).shape() == TensorShape({np, nf}),
                  errors::InvalidArgument("Invalid radial seed shape"));
    }
    if constexpr (mode == 2) {
      OP_REQUIRES(c, c->input(4).shape() == distances.shape(),
                  errors::InvalidArgument("Invalid radial direction shape"));
    }
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, mode == 0 || mode == 3 ? TensorShape({np, nf}) : distances.shape(), &output));
    double* out = output->flat<double>().data();
    double* dseed = nullptr;
    bool need_second = false;
    if constexpr (mode == 2 || mode == 3) {
      OP_REQUIRES_OK(c, c->allocate_output(1, TensorShape({np, nf}), &output));
      dseed = output->flat<double>().data();
      need_second = mode == 2 && c->output_required(0);
    }
    const double* seed = mode == 1 || mode == 2 ? c->input(3).flat<double>().data() : nullptr;
    const double* h = mode == 2 ? c->input(4).flat<double>().data() : nullptr;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double scale = std::pow(rc, 1.5);
    auto work = [&](int64_t begin, int64_t end) {
      std::vector<double> sinc(nf + 1), first(nf + 1), second(nf + 1);
      for (int64_t p = begin; p < end; ++p) {
        const double d = r[p];
        if constexpr (mode == 1 || mode == 2) out[p] = 0.;
        if (d >= rc || d < rin - delta) {
          if constexpr (mode == 0 || mode == 3) std::fill(out + p * nf, out + (p + 1) * nf, 0.);
          if constexpr (mode == 2 || mode == 3) std::fill(dseed + p * nf, dseed + (p + 1) * nf, 0.);
          continue;
        }
        double q = 1., q1 = 0., q2 = 0.;
        if (d <= rin) {
          const double x = 1. - 2. * (1. + (d - rin) / (delta + 1e-8));
          const double x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x4 * x;
          const double dx = -2. / (delta + 1e-8);
          const double f = 7.5 * (x / 4. - x3 / 6. + x5 / 20.);
          q = 1. - 0.5 * (1. + f);
          if constexpr (mode >= 1) q1 = -0.5 * 7.5 * (0.25 - x2 / 2. + x4 / 4.) * dx;
          if (need_second) q2 = -0.5 * 7.5 * (-x + x3) * dx * dx;
        }
        const double base_phase = d * pi / rc;
        // Keep the old near-zero path, where cancellation in analytic sinc
        // derivatives could amplify recurrence roundoff. Re-anchor every 16
        // frequencies, bounding drift for larger bases as well as WBC's 15.
        const bool use_trig = trig_recurrence && std::abs(base_phase) >= 0.02;
        double base_sin = 0., base_cos = 0., running_sin = 0., running_cos = 0.;
        if constexpr (trig_recurrence) {
          if (use_trig) {
            base_sin = std::sin(base_phase);
            base_cos = std::cos(base_phase);
            running_sin = base_sin;
            running_cos = base_cos;
          }
        }
        for (int64_t k = 0; k <= nf; ++k) {
          const double phase = d * (k + 1.) * pi / rc;
          const double frequency = (k + 1.) * pi / rc;
          if (std::abs(phase) < 1e-3) {
            // Analytic limits avoid cancellation and division by zero.
            const double z = phase * phase;
            sinc[k] = 1. + z * (-1./6. + z * (1./120. + z * (-1./5040. + z/362880.)));
            if constexpr (mode >= 1) {
              first[k] = phase * (-1./3. + z * (1./30. + z * (-1./840. + z/45360.))) * frequency;
            }
            if (need_second) second[k] = (-1./3. + z * (1./10. + z * (-1./168. + z/6480.))) * frequency * frequency;
          } else {
            double sn, cs = 0.;
            if constexpr (trig_recurrence) {
              if (use_trig) {
                if (k != 0) {
                  if (k % 16 == 0) {
                    running_sin = std::sin(phase);
                    running_cos = std::cos(phase);
                  } else {
                    const double next_sin = running_sin * base_cos + running_cos * base_sin;
                    running_cos = running_cos * base_cos - running_sin * base_sin;
                    running_sin = next_sin;
                  }
                }
                sn = running_sin;
                cs = running_cos;
              } else {
                sn = std::sin(phase);
                if constexpr (mode >= 1) cs = std::cos(phase);
              }
            } else {
              sn = std::sin(phase);
              if constexpr (mode >= 1) cs = std::cos(phase);
            }
            sinc[k] = sn / phase;
            if constexpr (mode >= 1) {
              const double df = (cs - sinc[k]) / phase;
              first[k] = df * frequency;
              if (need_second) second[k] = (-sn - 2. * df) / phase * frequency * frequency;
            }
          }
        }
        double prev = 0., prev1 = 0., prev2 = 0.;
        for (int64_t n = 0; n < nf; ++n) {
          const double factor = plan[n * 3], norm = plan[n * 3 + 1], mix = plan[n * 3 + 2];
          const double g = (factor * (sinc[n] + sinc[n + 1]) / scale + mix * prev) * norm;
          if constexpr (mode == 0) {
            out[p * nf + n] = g * q;
          } else {
            const double g1 = (factor * (first[n] + first[n + 1]) / scale + mix * prev1) * norm;
            const double derivative = g1 * q + g * q1;
            if constexpr (mode == 3) {
              out[p * nf + n] = g * q;
              dseed[p * nf + n] = derivative;
            }
            if constexpr (mode == 1) out[p] += seed[p * nf + n] * derivative;
            if constexpr (mode == 2) {
              dseed[p * nf + n] = h[p] * derivative;
              if (need_second) {
                const double g2 = (factor * (second[n] + second[n + 1]) / scale + mix * prev2) * norm;
                out[p] += seed[p * nf + n] * h[p] * (g2 * q + 2. * g1 * q1 + g * q2);
                prev2 = g2;
              }
            }
            prev1 = g1;
          }
          prev = g;
        }
      }
    };
    const auto* workers = c->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, np, nf * (mode == 0 ? 70 : 120), work);
  }
 private:
  bool trig_recurrence_;
};

REGISTER_KERNEL_BUILDER(Name("WbcRadial").Device(DEVICE_CPU), WbcRadialKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcRadialWithDerivative").Device(DEVICE_CPU), WbcRadialKernel<3>);
REGISTER_KERNEL_BUILDER(Name("WbcRadialGrad").Device(DEVICE_CPU), WbcRadialKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcRadialGradGrad").Device(DEVICE_CPU), WbcRadialKernel<2>);
