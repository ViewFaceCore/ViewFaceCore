#include <kernels/gpu/resize2d.h>
#include <core/tensor_builder.h>
#include <memory>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void Resize2d_ResizeImageLinear_kernel(const T *src_im, int src_width, int src_height,
                                                                 int channels, T *dst_im, int dst_width, int dst_height,
                                                                 int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            double lfx_scl = double(src_width) / dst_width;
            double lfy_scl = double(src_height) / dst_height;
            double bias_x = lfx_scl / 2 - 0.5;
            double bias_y = lfy_scl / 2 - 0.5;

            double lf_x_s = lfx_scl * n_x_d + bias_x;
            double lf_y_s = lfy_scl * n_y_d + bias_y;

            lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
            lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
            lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
            lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

            int n_x_s = int(lf_x_s);
            int n_y_s = int(lf_y_s);

            double lf_weight_x = lf_x_s - n_x_s;
            double lf_weight_y = lf_y_s - n_y_s;

            dst_im[index] = (T) ((1 - lf_weight_y) * (1 - lf_weight_x) *
                                 src_im[(n_y_s * src_width + n_x_s) * channels + c] +
                                 (1 - lf_weight_y) * lf_weight_x *
                                 src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
                                 lf_weight_y * (1 - lf_weight_x) *
                                 src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
                                 lf_weight_y * lf_weight_x *
                                 src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c]);


        }

#ifdef TS_USE_CUDA_FP16

        template<>
        __global__ void Resize2d_ResizeImageLinear_kernel<half>(const half *src_im, int src_width, int src_height,
                                                                int channels, half *dst_im, int dst_width,
                                                                int dst_height, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            double lfx_scl = double(src_width) / dst_width;
            double lfy_scl = double(src_height) / dst_height;
            double bias_x = lfx_scl / 2 - 0.5;
            double bias_y = lfy_scl / 2 - 0.5;

            double lf_x_s = lfx_scl * n_x_d + bias_x;
            double lf_y_s = lfy_scl * n_y_d + bias_y;

            lf_x_s = lf_x_s >= 0 ? lf_x_s : 0;
            lf_x_s = lf_x_s < src_width - 1 ? lf_x_s : src_width - 1 - 1e-5;
            lf_y_s = lf_y_s >= 0 ? lf_y_s : 0;
            lf_y_s = lf_y_s < src_height - 1 ? lf_y_s : src_height - 1 - 1e-5;

            int n_x_s = int(lf_x_s);
            int n_y_s = int(lf_y_s);

            half lf_weight_x = half(lf_x_s) - half(float(n_x_s));
            half lf_weight_y = half(lf_y_s) - half(float(n_y_s));

            half half_one = half(1.f);

            dst_im[index] = (half) ((half_one - lf_weight_y) * (half_one - lf_weight_x) *
                                    src_im[(n_y_s * src_width + n_x_s) * channels + c] +
                                    (half_one - lf_weight_y) * lf_weight_x *
                                    src_im[(n_y_s * src_width + n_x_s + 1) * channels + c] +
                                    lf_weight_y * (half_one - lf_weight_x) *
                                    src_im[((n_y_s + 1) * src_width + n_x_s) * channels + c] +
                                    lf_weight_y * lf_weight_x *
                                    src_im[((n_y_s + 1) * src_width + n_x_s + 1) * channels + c]);


        }

#endif


        template<typename T>
        static __global__ void Resize2d_ResizeImageCubic_kernel(const T *src_im, int src_width, int src_height,
                                                                int channels, T *dst_im, int dst_width, int dst_height,
                                                                int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int j = ntmp / nstep;
            ntmp = ntmp % nstep;

            int i = ntmp / channels;
            int k = ntmp % channels;

            double scale_x = (double) src_width / dst_width;
            double scale_y = (double) src_height / dst_height;

            int srcrows = src_width * channels;
            // int dstrows = dst_width * channels;

            double fy = (double) ((j + 0.5) * scale_y - 0.5);
            int sy = floor(fy);
            fy -= sy;

            if (sy < 1) {
                fy = 0;
                sy = 1;
            }

            if (sy >= src_height - 3) {
                fy = 0, sy = src_height - 3;
            }

            const double A = -0.75f;

            double coeffsY[4];
            coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
            coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
            coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
            coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

            double fx = (double) ((i + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 1) {
                fx = 0, sx = 1;
            }
            if (sx >= src_width - 3) {
                fx = 0, sx = src_width - 3;
            }

            double coeffsX[4];
            coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
            coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
            coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
            coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

            dst_im[index] = (T) ((
                    src_im[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3]));

        }

#ifdef TS_USE_CUDA_FP16

        template<>
        __global__ void Resize2d_ResizeImageCubic_kernel<half>(const half *src_im, int src_width, int src_height,
                                                               int channels, half *dst_im, int dst_width,
                                                               int dst_height, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int j = ntmp / nstep;
            ntmp = ntmp % nstep;

            int i = ntmp / channels;
            int k = ntmp % channels;

            double scale_x = (double) src_width / dst_width;
            double scale_y = (double) src_height / dst_height;

            int srcrows = src_width * channels;
            // int dstrows = dst_width * channels;

            double fy = (double) ((j + 0.5) * scale_y - 0.5);
            int sy = floor(fy);
            fy -= sy;

            if (sy < 1) {
                fy = 0;
                sy = 1;
            }

            if (sy >= src_height - 3) {
                fy = 0, sy = src_height - 3;
            }

            const double A = -0.75f;

            half coeffsY[4];
            coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
            coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
            coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
            coeffsY[3] = half(1.f) - coeffsY[0] - coeffsY[1] - coeffsY[2];

            double fx = (double) ((i + 0.5) * scale_x - 0.5);
            int sx = floor(fx);
            fx -= sx;

            if (sx < 1) {
                fx = 0, sx = 1;
            }
            if (sx >= src_width - 3) {
                fx = 0, sx = src_width - 3;
            }

            half coeffsX[4];
            coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
            coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
            coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
            coeffsX[3] = half(1.f) - coeffsX[0] - coeffsX[1] - coeffsX[2];

            dst_im[index] = (half) ((
                    src_im[(sy - 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx - 1) * channels + k] * coeffsX[0] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx) * channels + k] * coeffsX[1] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx + 1) * channels + k] * coeffsX[2] * coeffsY[3] +

                    src_im[(sy - 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[0] +
                    src_im[(sy) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[1] +
                    src_im[(sy + 1) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[2] +
                    src_im[(sy + 2) * srcrows + (sx + 2) * channels + k] * coeffsX[3] * coeffsY[3]));

        }

#endif

        template<typename T>
        static __global__ void Resize2d_ResizeNearest_kernel(const T *src_im, int src_width, int src_height,
                                                             int channels, T *dst_im, int dst_width, int dst_height,
                                                             int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            double lfx_scl = double(src_width) / dst_width;
            double lfy_scl = double(src_height) / dst_height;
            double bias_x = lfx_scl / 2 - 0.5;
            double bias_y = lfy_scl / 2 - 0.5;

            double lf_x_s = lfx_scl * n_x_d + bias_x;
            double lf_y_s = lfy_scl * n_y_d + bias_y;

            auto n_x_s = int(lf_x_s + 0.5);
            auto n_y_s = int(lf_y_s + 0.5);

            n_x_s = n_x_s >= 0 ? n_x_s : 0;
            n_x_s = n_x_s < src_width - 1 ? n_x_s : src_width - 1;
            n_y_s = n_y_s >= 0 ? n_y_s : 0;
            n_y_s = n_y_s < src_height - 1 ? n_y_s : src_height - 1;

            dst_im[index] = (T) src_im[(n_y_s * src_width + n_x_s) * channels + c];
        }

        template<typename T>
        static __global__ void Resize2d_ResizeHard_kernel(const T *src_im, int src_width, int src_height,
                                                          int channels, T *dst_im, int dst_width, int dst_height,
                                                          int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) {
                return;
            }

            int ntmp = index;

            int nstep = channels * dst_width;
            int n_y_d = ntmp / nstep;
            ntmp = ntmp % nstep;

            int n_x_d = ntmp / channels;
            int c = ntmp % channels;

            float lfx_scl = float(src_width) / dst_width;
            float lfy_scl = float(src_height) / dst_height;

            float lf_x_s = lfx_scl * n_x_d;
            float lf_y_s = lfy_scl * n_y_d;

            auto n_x_s = int(lf_x_s);
            auto n_y_s = int(lf_y_s);

            n_x_s = n_x_s >= 0 ? n_x_s : 0;
            n_x_s = n_x_s < src_width - 1 ? n_x_s : src_width - 1;
            n_y_s = n_y_s >= 0 ? n_y_s : 0;
            n_y_s = n_y_s < src_height - 1 ? n_y_s : src_height - 1;

            dst_im[index] = (T) src_im[(n_y_s * src_width + n_x_s) * channels + c];
        }

        template<typename T>
        static inline void resize_linear(const Tensor *x, Tensor *y, int x_height, int x_width,
                                         int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                         int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;

            int ncount = y_height * y_width * channels;

            RUN_KERNEL(Resize2d_ResizeImageLinear_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       psrc, x_width, x_height, channels, pdst, y_width, y_height, ncount);

        }


        template<typename T>
        static inline void resize_cubic(const Tensor *x, Tensor *y, int x_height, int x_width,
                                        int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                        int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;

            int ncount = y_height * y_width * channels;

            RUN_KERNEL(Resize2d_ResizeImageCubic_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       psrc, x_width, x_height, channels, pdst, y_width, y_height, ncount);
        }


        template<typename T>
        static inline void resize_nearest(const Tensor *x, Tensor *y, int x_height, int x_width,
                                          int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                          int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;

            int ncount = y_height * y_width * channels;

            RUN_KERNEL(Resize2d_ResizeNearest_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       psrc, x_width, x_height, channels, pdst, y_width, y_height, ncount);
        }


        template<typename T>
        static inline void resize_hard(const Tensor *x, Tensor *y, int x_height, int x_width,
                                       int y_height, int y_width, unsigned int x_offset, unsigned int y_offset,
                                       int channels) {

            const T *psrc = x->data<T>() + x_offset;
            T *pdst = y->data<T>() + y_offset;

            int ncount = y_height * y_width * channels;

            RUN_KERNEL(Resize2d_ResizeHard_kernel<T>, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       psrc, x_width, x_height, channels, pdst, y_width, y_height, ncount);
        }

        template<typename T>
        static inline void batch_resize_linear(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                               int y_height, int y_width,
                                               unsigned int x_batch_step, unsigned int y_batch_step,
                                               int channels) {
            for (int k = 0; k < number; k++) {
                resize_linear<T>(x, y, x_height, x_width, y_height, y_width,
                                 k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_hard(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                             int y_height, int y_width,
                                             unsigned int x_batch_step, unsigned int y_batch_step,
                                             int channels) {
            for (int k = 0; k < number; k++) {
                resize_hard<T>(x, y, x_height, x_width, y_height, y_width,
                               k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_cubic(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                              int y_height, int y_width,
                                              unsigned int x_batch_step, unsigned int y_batch_step,
                                              int channels) {
            for (int k = 0; k < number; k++) {
                resize_cubic<T>(x, y, x_height, x_width, y_height, y_width,
                                k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static inline void batch_resize_nearest(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                                int y_height, int y_width,
                                                unsigned int x_batch_step, unsigned int y_batch_step,
                                                int channels) {
            for (int k = 0; k < number; k++) {
                resize_nearest<T>(x, y, x_height, x_width, y_height, y_width,
                                  k * x_batch_step, k * y_batch_step, channels);
            }
        }

        template<typename T>
        static void batch_resize(int number, const Tensor *x, Tensor *y, int x_height, int x_width,
                                 int y_height, int y_width,
                                 unsigned int x_batch_step, unsigned int y_batch_step,
                                 int channels, Resize2DType type) {
            if (type == Resize2DType::LINEAR) {
                batch_resize_linear<T>(number, x, y,
                                       x_height, x_width,
                                       y_height, y_width,
                                       x_batch_step, y_batch_step, channels);

            } else if (type == Resize2DType::CUBIC) {
                batch_resize_cubic<T>(number, x, y,
                                      x_height, x_width,
                                      y_height, y_width,
                                      x_batch_step, y_batch_step, channels);
            } else if (type == Resize2DType::HARD) {
                batch_resize_hard<T>(number, x, y,
                                     x_height, x_width,
                                     y_height, y_width,
                                     x_batch_step, y_batch_step, channels);
            } else {
                batch_resize_nearest<T>(number, x, y,
                                        x_height, x_width,
                                        y_height, y_width,
                                        x_batch_step, y_batch_step, channels);
            }
        }

        void Resize2D::resize2d(const Tensor &x, int i, Resize2DType type, Tensor &out) {
            auto &output_shape = out.sizes();

            int y_height = out.size(i);
            int y_width = out.size(i + 1);
            int x_height = x.size(i);
            int x_width = x.size(i + 1);

            int number, channels;
            number = channels = 1;

            for (int k = 0; k < i; k++) {
                number *= output_shape[k];
            }

            for (int k = i + 2; k < output_shape.size(); k++) {
                channels *= output_shape[k];
            }

            int y_batch_step = channels * y_height * y_width;
            int x_batch_step = channels * x_height * x_width;

            const Tensor *input = &x;
            Tensor *output = &out;
            ts::DTYPE dtype = output->dtype();

            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { batch_resize<TYPE>( \
                        number, input, output, \
                        x_height, x_width, \
                        y_height, y_width, \
                        x_batch_step, y_batch_step, channels, type); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
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
TS_REGISTER_OPERATOR(Resize2D, GPU, name::layer::resize2d())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Resize2D, GPU, name::layer::resize2d())
#endif
