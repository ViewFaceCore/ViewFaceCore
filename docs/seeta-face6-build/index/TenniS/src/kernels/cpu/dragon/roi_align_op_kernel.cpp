//
// Created by kier on 2019/9/7.
//

#include "op_kernel.h"
#include "core/ieee754_float.h"

#include "kernels/common/third/dragon.h"

namespace ts {

    namespace dragon {

        namespace kernel {

/*! ROIAlign <T = float32, Device = CPU> */

            template<typename T>
            T _ROIAlignInterpolate(
                    const T *Xdata,
                    const int height,
                    const int width,
                    T y,
                    T x) {
                if (y < -1.0 || y > height || x < -1.0 || x > width) return 0;
                if (y <= 0) y = 0;
                if (x <= 0) x = 0;

                int y_low = (int) y;
                int x_low = (int) x;
                int y_high;
                int x_high;

                if (y_low >= height - 1) {
                    y_high = y_low = height - 1;
                    y = (T) y_low;
                } else {
                    y_high = y_low + 1;
                }

                if (x_low >= width - 1) {
                    x_high = x_low = width - 1;
                    x = (T) x_low;
                } else {
                    x_high = x_low + 1;
                }

                T ly = y - y_low;
                T lx = x - x_low;
                T hy = (T) 1 - ly, hx = (T) 1 - lx;
                T v1 = Xdata[y_low * width + x_low];
                T v2 = Xdata[y_low * width + x_high];
                T v3 = Xdata[y_high * width + x_low];
                T v4 = Xdata[y_high * width + x_high];
                T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
                T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
                return val;
            }

            template<>
            void ROIAlign<float, CPUContext>(
                    const int C,
                    const int H,
                    const int W,
                    const int pool_h,
                    const int pool_w,
                    const int num_rois,
                    const float spatial_scale,
                    const int sampling_ratio,
                    const float *x,
                    const float *rois,
                    float *y,
                    CPUContext *ctx) {
                const int64_t X_offset = H * W, Y_offset = pool_h * pool_w;
                const int64_t x_offset = C * X_offset, y_offset = C * Y_offset;

                for (int n = 0; n < num_rois; ++n) {
                    auto *R = rois + n * 5;
                    int roi_batch_ind = (int) R[0];
                    auto *Y = y + n * y_offset;

                    if (roi_batch_ind < 0) {
                        std::memset(Y, 0, sizeof(float) * y_offset);
                        continue;
                    }

                    float roi_start_w = R[1] * spatial_scale;
                    float roi_start_h = R[2] * spatial_scale;
                    float roi_end_w = R[3] * spatial_scale;
                    float roi_end_h = R[4] * spatial_scale;

                    float roi_width = std::max(roi_end_w - roi_start_w, 1.f);
                    float roi_height = std::max(roi_end_h - roi_start_h, 1.f);
                    float bin_size_h = (float) roi_height / (float) pool_h;
                    float bin_size_w = (float) roi_width / (float) pool_w;

                    int roi_bin_grid_h = (sampling_ratio > 0) ?
                                         sampling_ratio : (int) ceil(roi_height / pool_h);
                    int roi_bin_grid_w = (sampling_ratio > 0) ?
                                         sampling_ratio : (int) ceil(roi_width / pool_w);

                    const float num_bin_grids = (float) roi_bin_grid_h * roi_bin_grid_w;
                    const float *X = x + roi_batch_ind * x_offset;

                    for (int c = 0; c < C; ++c) {
                        for (int ph = 0; ph < pool_h; ++ph) {
                            for (int pw = 0; pw < pool_w; ++pw) {
                                float output_val = 0.f;
                                const int pool_idx = ph * pool_w + pw;
                                for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                                    const float y = roi_start_h + ph * bin_size_h +
                                                    static_cast<float>(iy + .5f) * bin_size_h /
                                                    static_cast<float>(roi_bin_grid_h);
                                    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                                        const float x = roi_start_w + pw * bin_size_w +
                                                        static_cast<float>(ix + .5f) * bin_size_w /
                                                        static_cast<float>(roi_bin_grid_w);
                                        output_val += _ROIAlignInterpolate<float>(X, H, W, y, x);
                                    }  // End ix
                                }  // End iy
                                output_val /= num_bin_grids;
                                Y[pool_idx] = output_val;
                            }  // End pw
                        }  // End ph
                        // Offset according to C
                        X += X_offset;
                        Y += Y_offset;
                    }  // End c
                }  // End n
            }

/*! ROIAlign <T = float16, Device = CPU> */

            template<>
            void ROIAlign<float16, CPUContext>(
                    const int C,
                    const int H,
                    const int W,
                    const int pool_h,
                    const int pool_w,
                    const int num_rois,
                    const float spatial_scale,
                    const int sampling_ratio,
                    const float16 *x,
                    const float *rois,
                    float16 *y,
                    CPUContext *ctx) {
                CPU_FP16_NOT_SUPPORTED;
            }

/*! ROIAlignGrad <T = float32, Device = CPU> */

        }  // namespace kernel

    }  // namespace dragon

} // namespace ts