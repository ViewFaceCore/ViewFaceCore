//
// Created by kier on 19-4-17.
//

#ifndef TENSORSTACK_KERNELS_CPU_DCN_DCN_V2_H
#define TENSORSTACK_KERNELS_CPU_DCN_DCN_V2_H

#include "core/tensor.h"

/**
 * @see [DCNv2](https://github.com/CharlesShang/DCNv2)
 */

namespace ts {
    Tensor dcn_v2_cpu_forward(const Tensor &input,
                   const Tensor &weight,
                   const Tensor &bias,
                   const Tensor &offset,
                   const Tensor &mask,
                   const int kernel_h,
                   const int kernel_w,
                   const int stride_h,
                   const int stride_w,
                   const int pad_h,
                   const int pad_w,
                   const int dilation_h,
                   const int dilation_w,
                   const int deformable_group,
                   Tensor *buffer_output = nullptr);
}

#endif //TENSORSTACK_KERNELS_CPU_DCN_DCN_V2_H
