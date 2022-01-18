#ifndef TENSORSTACK_KERNELS_CPU_IM2COL_H
#define TENSORSTACK_KERNELS_CPU_IM2COL_H

#include <vector>

namespace ts {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h_top,const int pad_h_bottom, const int pad_w_left,const int pad_w_right, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_col, const Dtype padding_value);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    //const int pad_h, const int pad_w, const int stride_h,
    const int pad_h_top,const int pad_h_bottom, const int pad_w_left,const int pad_w_right, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);


};

#endif  // TENSORSTACK_KERNELS_CPU_IM2COL_H
