#include "dcn_v2_im2col_cpu.h"

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <cmath>

#include <kernels/common/openmp.h>

#define CPU_KERNEL_LOOP(i, n)                          \
  for (int i = 0; i < (n); ++i)

static float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                           const int height, const int width, float h, float w) {
    int h_low = int(floor(h));
    int w_low = int(floor(w));
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

static void modulated_deformable_im2col_gpu_kernel(const int n,
                                            const float *data_im, const float *data_offset, const float *data_mask,
                                            const int height, const int width, const int kernel_h, const int kernel_w,
                                            const int pad_h, const int pad_w,
                                            const int stride_h, const int stride_w,
                                            const int dilation_h, const int dilation_w,
                                            const int channel_per_deformable_group,
                                            const int batch_size, const int num_channels, const int deformable_group,
                                            const int height_col, const int width_col,
                                            float *data_col)
{
    // launch channels * batch_size * height_col * width_col cores


    // add openmp
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(ts::openmp_threads())
#endif
    CPU_KERNEL_LOOP(index, n)
    {
        // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
        // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis

        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        // const int b_col = (index / width_col / height_col) % batch_size;
        const int b_col = (index / width_col / height_col / num_channels) % batch_size;
        // const int c_im = (index / width_col / height_col) / batch_size;
        const int c_im = (index / width_col / height_col) % num_channels;
        // const int c_col = c_im * kernel_h * kernel_w;
        const int c_col = c_im * kernel_h * kernel_w;

        // compute deformable group index
        const int deformable_group_index = c_im / channel_per_deformable_group;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;

        //  float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
        //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
        const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
        const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

        const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
                const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                const float offset_h = data_offset_ptr[data_offset_h_ptr];
                const float offset_w = data_offset_ptr[data_offset_w_ptr];
                const float mask = data_mask_ptr[data_mask_hw_ptr];
                float val = static_cast<float>(0);
                const float h_im = h_in + i * dilation_h + offset_h;
                const float w_im = w_in + j * dilation_w + offset_w;
                //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
                {
                    //const float map_h = i * dilation_h + offset_h;
                    //const float map_w = j * dilation_w + offset_w;
                    //const int cur_height = height - h_in;
                    //const int cur_width = width - w_in;
                    //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
                    val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
                }
                *data_col_ptr = val * mask;
                // data_col_ptr += batch_size * height_col * width_col;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void modulated_deformable_im2col_cpu(const float* data_im, const float* data_offset, const float* data_mask,
                                     const int batch_size, const int channels, const int height_im, const int width_im,
                                     const int height_col, const int width_col, const int kernel_h, const int kernel_w,
                                     const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                     const int dilation_h, const int dilation_w,
                                     const int deformable_group, float* data_col) {
    // num_axes should be smaller than block size
    const int channel_per_deformable_group = channels / deformable_group;
    const int num_kernels = channels * batch_size * height_col * width_col;

    modulated_deformable_im2col_gpu_kernel(
            num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
            batch_size, channels, deformable_group, height_col, width_col, data_col);
}