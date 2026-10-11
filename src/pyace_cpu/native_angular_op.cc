// Sparse Cartesian polynomial form of the upstream complex spherical harmonics.
// Float64 values and exact first/second derivatives, including off-sphere inputs.
#include <algorithm>
#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using shape_inference::InferenceContext;
#define ANGULAR_INPUTS \
    .Input("xyz: double").Input("row_ptr: int32") \
    .Input("powers: int32").Input("weights: double")

REGISTER_OP("WbcAngular") ANGULAR_INPUTS .Output("y: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::DimensionHandle ny;
      TF_RETURN_IF_ERROR(c->Subtract(c->Dim(c->input(1), 0), 1, &ny));
      c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), ny));
      return absl::OkStatus();
    });
REGISTER_OP("WbcAngularGrad") ANGULAR_INPUTS .Input("seed: double").Output("dx: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });
REGISTER_OP("WbcAngularGradGrad") ANGULAR_INPUTS
    .Input("seed: double").Input("direction: double")
    .Output("dx: double").Output("dseed: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(4));
      return absl::OkStatus();
    });

REGISTER_OP("WbcAngularSplit") ANGULAR_INPUTS
    .Output("yr: double").Output("yi: double")
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::DimensionHandle ny, nm;
      TF_RETURN_IF_ERROR(c->Subtract(c->Dim(c->input(1), 0), 1, &ny));
      TF_RETURN_IF_ERROR(c->Divide(ny, 2, true, &nm));
      for (int k = 0; k < 2; ++k) c->set_output(k, c->Matrix(c->Dim(c->input(0), 0), nm));
      return absl::OkStatus();
    });
REGISTER_OP("WbcAngularSplitGrad") ANGULAR_INPUTS
    .Input("sr: double").Input("si: double").Output("dx: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });
REGISTER_OP("WbcAngularSplitGradGrad") ANGULAR_INPUTS
    .Input("sr: double").Input("si: double").Input("direction: double")
    .Output("dx: double").Output("dsr: double").Output("dsi: double")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(4));
      c->set_output(2, c->input(5));
      return absl::OkStatus();
    });

template <int mode, bool split = false>
class WbcAngularKernel : public OpKernel {
 public:
  explicit WbcAngularKernel(OpKernelConstruction* c) : OpKernel(c) {}
  void Compute(OpKernelContext* c) override {
    const Tensor& xyz = c->input(0);
    OP_REQUIRES(c, xyz.dims() == 2 && xyz.dim_size(1) == 3 &&
                c->input(1).dims() == 1 && c->input(2).dims() == 2 &&
                c->input(2).dim_size(1) == 3 && c->input(3).dims() == 1,
                errors::InvalidArgument("Invalid angular input shapes"));
    const int64_t np = xyz.dim_size(0), ny = c->input(1).NumElements() - 1;
    const int64_t nout = split ? ny / 2 : ny;
    const int64_t nt = c->input(2).dim_size(0);
    OP_REQUIRES(c, ny >= 1 && (!split || ny % 2 == 0) && c->input(3).NumElements() == nt,
                errors::InvalidArgument("Invalid angular plan lengths"));
    const int32* row = c->input(1).flat<int32>().data();
    const int32* powers = c->input(2).flat<int32>().data();
    const double* weights = c->input(3).flat<double>().data();
    OP_REQUIRES(c, row[0] == 0 && row[ny] == nt,
                errors::InvalidArgument("Invalid angular row boundaries"));
    for (int64_t y = 0; y < ny; ++y) {
      OP_REQUIRES(c, row[y] >= 0 && row[y] <= row[y + 1] && row[y + 1] <= nt,
                  errors::InvalidArgument("Invalid angular row order"));
    }
    int max_power = 0;
    for (int64_t t = 0; t < nt * 3; ++t) {
      OP_REQUIRES(c, powers[t] >= 0 && powers[t] <= 32,
                  errors::InvalidArgument("Angular polynomial power out of supported range"));
      max_power = std::max(max_power, powers[t]);
    }
    if constexpr (mode >= 1) {
      OP_REQUIRES(c, c->input(4).shape() == TensorShape({np, nout}),
                  errors::InvalidArgument("Invalid angular seed shape"));
      if constexpr (split) {
        OP_REQUIRES(c, c->input(5).shape() == TensorShape({np, nout}),
                    errors::InvalidArgument("Invalid angular imaginary seed shape"));
      }
    }
    if constexpr (mode == 2) {
      OP_REQUIRES(c, c->input(split ? 6 : 5).shape() == xyz.shape(),
                  errors::InvalidArgument("Invalid angular direction shape"));
    }
    const double* x = xyz.flat<double>().data();
    const double* seed = mode >= 1 ? c->input(4).flat<double>().data() : nullptr;
    const double* seed_i = mode >= 1 && split ? c->input(5).flat<double>().data() : nullptr;
    const double* h = mode == 2 ? c->input(split ? 6 : 5).flat<double>().data() : nullptr;
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, mode == 0 ? TensorShape({np, nout}) : xyz.shape(), &output));
    double* out = output->flat<double>().data();
    double* dseed = nullptr;
    double* imaginary = nullptr;
    if constexpr (split && mode == 0) {
      OP_REQUIRES_OK(c, c->allocate_output(1, TensorShape({np, nout}), &output));
      imaginary = output->flat<double>().data();
    }
    bool need_hessian = true;
    if constexpr (mode == 2) {
      OP_REQUIRES_OK(c, c->allocate_output(1, TensorShape({np, nout}), &output));
      dseed = output->flat<double>().data();
      if constexpr (split) {
        OP_REQUIRES_OK(c, c->allocate_output(2, TensorShape({np, nout}), &output));
        imaginary = output->flat<double>().data();
      }
      need_hessian = c->output_required(0);
    }
    auto work = [&](int64_t begin, int64_t end) {
      double xp[3][33];
      for (int64_t p = begin; p < end; ++p) {
        for (int j = 0; j < 3; ++j) {
          xp[j][0] = 1.;
          for (int k = 1; k <= max_power; ++k) xp[j][k] = xp[j][k - 1] * x[p * 3 + j];
          if constexpr (mode >= 1) out[p * 3 + j] = 0.;
        }
        for (int64_t y = 0; y < ny; ++y) {
          const bool imag = split && y >= nout;
          const int64_t index = p * nout + (imag ? y - nout : y);
          const double seed_value = mode >= 1 ? (imag ? seed_i[index] : seed[index]) : 0.;
          double value = 0., gradient[3] = {}, hessian_direction[3] = {};
          for (int64_t t = row[y]; t < row[y + 1]; ++t) {
            const int32* e = powers + t * 3;
            const double w = weights[t];
            if constexpr (mode == 0) {
              value += w * xp[0][e[0]] * xp[1][e[1]] * xp[2][e[2]];
            } else {
              for (int j = 0; j < 3; ++j) {
                if (!e[j]) continue;
                double d = w * e[j];
                for (int a = 0; a < 3; ++a) d *= xp[a][e[a] - (a == j)];
                gradient[j] += d;
                if constexpr (mode == 2) {
                  if (!need_hessian) continue;
                  for (int k = 0; k < 3; ++k) {
                    if (e[k] <= (k == j)) continue;
                    double d2 = w * e[j] * (e[k] - (k == j));
                    for (int a = 0; a < 3; ++a) d2 *= xp[a][e[a] - (a == j) - (a == k)];
                    hessian_direction[j] += d2 * h[p * 3 + k];
                  }
                }
              }
            }
          }
          if constexpr (mode == 0) {
            (imag ? imaginary : out)[index] = value;
          } else if constexpr (mode == 1) {
            for (int j = 0; j < 3; ++j) out[p * 3 + j] += seed_value * gradient[j];
          } else {
            double direction = 0.;
            for (int j = 0; j < 3; ++j) {
              direction += gradient[j] * h[p * 3 + j];
              if (need_hessian) out[p * 3 + j] += seed_value * hessian_direction[j];
            }
            (imag ? imaginary : dseed)[index] = direction;
          }
        }
      }
    };
    const auto* workers = c->device()->tensorflow_cpu_worker_threads();
    Shard(workers->num_threads, workers->workers, np, std::max<int64_t>(1, nt * (mode == 0 ? 8 : 35)), work);
  }
};

REGISTER_KERNEL_BUILDER(Name("WbcAngular").Device(DEVICE_CPU), WbcAngularKernel<0>);
REGISTER_KERNEL_BUILDER(Name("WbcAngularGrad").Device(DEVICE_CPU), WbcAngularKernel<1>);
REGISTER_KERNEL_BUILDER(Name("WbcAngularGradGrad").Device(DEVICE_CPU), WbcAngularKernel<2>);
REGISTER_KERNEL_BUILDER(Name("WbcAngularSplit").Device(DEVICE_CPU), WbcAngularKernel<0, true>);
REGISTER_KERNEL_BUILDER(Name("WbcAngularSplitGrad").Device(DEVICE_CPU), WbcAngularKernel<1, true>);
REGISTER_KERNEL_BUILDER(Name("WbcAngularSplitGradGrad").Device(DEVICE_CPU), WbcAngularKernel<2, true>);
