#include <kernels/gpu/affine_sample2d.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "kernels/gpu/gpu_kernel.h"


namespace ts {
    namespace gpu {
        template<typename TO, typename FROM>
        static __device__ TO clamp(FROM MIN, FROM MAX, FROM from) {
            return TO(max(MIN, min(MAX, from)));
        }

        template<typename T>
        static __global__ void
        affine_sample2d_linear_kernel(const T *psrc, T *pdst, int size, int x_height, int x_width,
                                      int y_height, int y_width,
                //unsigned int x_offset, unsigned int y_offset,
                                      int channels, float rz00, float rz01, float rz02, float rz10,
                                      float rz11, float rz12, float rz20, float rz21, float rz22,
                                      double data_min, double data_max,
                                      base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * y_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            auto curx = rz00 * n_x_d + rz01 * n_y_d + rz02 * 1;
            auto cury = rz10 * n_x_d + rz11 * n_y_d + rz12 * 1;

            double lf_x_s = curx;
            double lf_y_s = cury;

            auto inner = lf_x_s >= 0 && lf_x_s < x_width - 1 &&
                         lf_y_s >= 0 && lf_y_s < x_height - 1;

            if (!inner && outer_mode == base::AffineOuterMode::VALUE) {
                pdst[index] = outer_value;
                return;
            }

            lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
            lf_x_s = lf_x_s < x_width - 1 ? lf_x_s : x_width - 1 - 1e-5;
            lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
            lf_y_s = lf_y_s < x_height - 1 ? lf_y_s : x_height - 1 - 1e-5;

            int n_x_s = int(lf_x_s);
            int n_y_s = int(lf_y_s);

            double lf_weight_x = lf_x_s - n_x_s;
            double lf_weight_y = lf_y_s - n_y_s;

            pdst[index] = clamp<T, double>(data_min, data_max, (
                    (1 - lf_weight_y) * (1 - lf_weight_x) *
                    psrc[(n_y_s * x_width + n_x_s) * channels + c] +
                    (1 - lf_weight_y) * lf_weight_x *
                    psrc[(n_y_s * x_width + n_x_s + 1) * channels + c] +
                    lf_weight_y * (1 - lf_weight_x) *
                    psrc[((n_y_s + 1) * x_width + n_x_s) * channels + c] +
                    lf_weight_y * lf_weight_x *
                    psrc[((n_y_s + 1) * x_width + n_x_s + 1) * channels + c]));

        }

        template<typename T>
        static __global__ void
        affine_sample2d_nearest_kernel(const T *psrc, T *pdst, int size, int x_height, int x_width,
                                       int y_height, int y_width,
                //unsigned int x_offset, unsigned int y_offset,
                                       int channels, float rz00, float rz01, float rz02, float rz10,
                                       float rz11, float rz12, float rz20, float rz21, float rz22,
                                       double data_min, double data_max,
                                       base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * y_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            auto curx = rz00 * n_x_d + rz01 * n_y_d + rz02 * 1;
            auto cury = rz10 * n_x_d + rz11 * n_y_d + rz12 * 1;

            double lf_x_s = curx;
            double lf_y_s = cury;

            auto n_x_s = int(lf_x_s + 0.5);
            auto n_y_s = int(lf_y_s + 0.5);

            auto inner = n_x_s >= 0 && n_x_s < x_width - 1 &&
                         n_y_s >= 0 && n_y_s < x_height - 1;

            if (!inner && outer_mode == base::AffineOuterMode::VALUE) {
                pdst[index] = outer_value;
                return;
            }

            n_x_s = n_x_s >= 0 ? n_x_s : 0;
            n_x_s = n_x_s < x_width - 1 ? n_x_s : x_width - 1;
            n_y_s = n_y_s >= 0 ? n_y_s : 0;
            n_y_s = n_y_s < x_height - 1 ? n_y_s : x_height - 1;

            pdst[index] = clamp<T, double>(
                    data_min, data_max,
                    psrc[(n_y_s * x_width + n_x_s) * channels + c]);

        }

        template<typename T>
        static __global__ void
        affine_sample2d_hard_kernel(const T *psrc, T *pdst, int size, int x_height, int x_width,
                                    int y_height, int y_width,
                //unsigned int x_offset, unsigned int y_offset,
                                    int channels, float rz00, float rz01, float rz02, float rz10,
                                    float rz11, float rz12, float rz20, float rz21, float rz22,
                                    double data_min, double data_max,
                                    base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * y_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            auto curx = rz00 * n_x_d + rz01 * n_y_d + rz02 * 1;
            auto cury = rz10 * n_x_d + rz11 * n_y_d + rz12 * 1;

            double lf_x_s = curx;
            double lf_y_s = cury;

            auto n_x_s = int(lf_x_s);
            auto n_y_s = int(lf_y_s);

            auto inner = n_x_s >= 0 && n_x_s < x_width - 1 &&
                         n_y_s >= 0 && n_y_s < x_height - 1;

            if (!inner && outer_mode == base::AffineOuterMode::VALUE) {
                pdst[index] = outer_value;
                return;
            }

            n_x_s = n_x_s >= 0 ? n_x_s : 0;
            n_x_s = n_x_s < x_width - 1 ? n_x_s : x_width - 1;
            n_y_s = n_y_s >= 0 ? n_y_s : 0;
            n_y_s = n_y_s < x_height - 1 ? n_y_s : x_height - 1;

            pdst[index] = clamp<T, double>(
                    data_min, data_max,
                    psrc[(n_y_s * x_width + n_x_s) * channels + c]);

        }

        template<typename T>
        static __global__ void affine_sample2d_cubic_kernel(const T *psrc, T *pdst, int size, int x_height, int x_width,
                                                            int y_height, int y_width,
                //unsigned int x_offset, unsigned int y_offset,
                                                            int channels, float rz00, float rz01, float rz02,
                                                            float rz10,
                                                            float rz11, float rz12, float rz20, float rz21, float rz22,
                                                            double data_min, double data_max,
                                                            base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * y_width;
            int j = ntmp / nstep;
            ntmp = ntmp % nstep;

            int i = ntmp / channels;
            int k = ntmp % channels;

            int srcrows = x_width * channels;

            double fx = rz00 * i + rz01 * j + rz02 * 1;
            double fy = rz10 * i + rz11 * j + rz12 * 1;

            const double A = -0.75f;

            int sy = floor(fy);
            fy -= sy;

            int sx = floor(fx);
            fx -= sx;

            auto outter = sy < 1 || sy >= x_height - 3 || sx < 1 || sx >= x_width - 3;

            if (outter && outer_mode == base::AffineOuterMode::VALUE) {
                pdst[index] = outer_value;
                return;
            }

            if (sy < 1) {
                fy = 0;
                sy = 1;
            }
            if (sy >= x_height - 3) {
                fy = 0;
                sy = x_height - 3;
            }
            if (sx < 1) {
                fx = 0;
                sx = 1;
            }
            if (sx >= x_width - 3) {
                fx = 0;
                sx = x_width - 3;
            }

            double coeffsY[4];
            coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
            coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
            coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
            coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

            double coeffsX[4];
            coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
            coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
            coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
            coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

            pdst[index] = clamp<T, double>(data_min, data_max, ((
                    psrc[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
                    psrc[(sy) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
                    psrc[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
                    psrc[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

                    psrc[(sy - 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                    psrc[(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                    psrc[(sy + 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                    psrc[(sy + 2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                    psrc[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
                    psrc[(sy) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
                    psrc[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
                    psrc[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

                    psrc[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
                    psrc[(sy) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
                    psrc[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
                    psrc[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3])));

        }


        template<typename T>
        static void batch_affine_sample2d(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                          int y_height, int y_width,
                                          unsigned int x_batch_step, unsigned int y_batch_step,
                                          int channels, Affine_Sample2DType type, float rz00, float rz01, float rz02,
                                          float rz10,
                                          float rz11, float rz12, float rz20, float rz21, float rz22,
                                          base::AffineOuterMode outer_mode, T outer_value = T(0)) {
            constexpr auto MAX = double(std::numeric_limits<T>::max());
            constexpr auto MIN = double(std::numeric_limits<T>::lowest());

            int ncount = y_height * y_width * channels;

            if (type == Affine_Sample2DType::CUBIC) {
                for (int k = 0; k < number; k++) {

                    const T *psrc = x->data<T>() + k * x_batch_step;
                    T *pdst = y->data<T>() + k * y_batch_step;
                    RUN_KERNEL(affine_sample2d_cubic_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                               psrc, pdst, ncount, x_height, x_width, y_height, y_width, channels,
                               rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22,
                               MIN, MAX,
                               outer_mode, outer_value);
                }
            } else if (type == Affine_Sample2DType::NEAREST) {

                for (int k = 0; k < number; k++) {

                    const T *psrc = x->data<T>() + k * x_batch_step;
                    T *pdst = y->data<T>() + k * y_batch_step;

                    RUN_KERNEL(affine_sample2d_nearest_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                               psrc, pdst, ncount, x_height, x_width, y_height, y_width, channels,
                               rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22,
                               MIN, MAX,
                               outer_mode, outer_value);
                }
            } else if (type == Affine_Sample2DType::HARD) {

                for (int k = 0; k < number; k++) {

                    const T *psrc = x->data<T>() + k * x_batch_step;
                    T *pdst = y->data<T>() + k * y_batch_step;

                    RUN_KERNEL(affine_sample2d_hard_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                               psrc, pdst, ncount, x_height, x_width, y_height, y_width, channels,
                               rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22,
                               MIN, MAX,
                               outer_mode, outer_value);
                }
            } else { //LINEAR
                for (int k = 0; k < number; k++) {

                    const T *psrc = x->data<T>() + k * x_batch_step;
                    T *pdst = y->data<T>() + k * y_batch_step;

                    RUN_KERNEL(affine_sample2d_linear_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                               psrc, pdst, ncount, x_height, x_width, y_height, y_width, channels,
                               rz00, rz01, rz02, rz10, rz11, rz12, rz20, rz21, rz22,
                               MIN, MAX,
                               outer_mode, outer_value);
                }
            }
        }

        void Affine_Sample2D::affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10,
                                                float rz11, float rz12, float rz20, float rz21, float rz22,
                                                Affine_Sample2DType type, int dim,
                                                base::AffineOuterMode outer_mode, float outer_value,
                                                Tensor &out) {

            auto &output_shape = out.sizes();

            int y_height = out.size(dim);
            int y_width = out.size(dim + 1);
            int x_height = x.size(dim);
            int x_width = x.size(dim + 1);

            int number, channels;
            number = channels = 1;

            for (int k = 0; k < dim; k++) {
                number *= output_shape[k];
            }

            for (int k = dim + 2; k < output_shape.size(); k++) {
                channels *= output_shape[k];
            }

            int y_batch_step = channels * y_height * y_width;
            int x_batch_step = channels * x_height * x_width;

            const Tensor *input = &x;
            Tensor *output = &out;
            ts::DTYPE dtype = output->dtype();

            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { batch_affine_sample2d<TYPE>( \
                        number, input, output, \
                        x_height, x_width, \
                        y_height, y_width, \
                        x_batch_step, y_batch_step, channels, type, rz00,rz01,rz02,rz10,rz11,rz12,rz20,rz21,rz22, \
                        outer_mode, TYPE(outer_value)); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }

        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Affine_Sample2D, GPU, name::layer::affine_sample2d())
