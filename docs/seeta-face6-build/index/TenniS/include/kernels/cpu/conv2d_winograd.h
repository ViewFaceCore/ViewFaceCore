#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_winograd.h"

namespace ts {
    namespace cpu{
        class Conv2DWinograd : public OperatorOnCPU<base::Conv2DWinograd> {
        public:
            using self = Conv2DWinograd;
            using supper = base::Conv2DWinograd;

            void conv2d_tranform_kernel(WinogradConv2DMode  winograd_mode, const Tensor &kernel, Tensor &kernel_transformed);

            void conv2d_winograd(const Tensor &x, WinogradConv2DMode winograd_mode, const Padding2D &padding, float padding_value,
                const Tensor &w, Conv2DFormat format, Tensor &out, bool kernel_transformed);
//            void conv2d_winograd(const Tensor &x, WinogradConv2DMode winograd_mode,
//                const Tensor &w, Conv2DFormat format, Tensor &out, Stack &stack);
        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_H