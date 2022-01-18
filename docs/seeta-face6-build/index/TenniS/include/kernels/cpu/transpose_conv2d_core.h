#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_TRANSPOSE_CORE_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_TRANSPOSE_CORE_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_transpose_core.h"


namespace ts {
    namespace cpu {
        class Conv2DTransposeCore : public base::Conv2DTransposeCore {
        public:
            using self = Conv2DTransposeCore;
            using supper = base::Conv2DTransposeCore;

            Conv2DTransposeCore() = default;

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override;
        };
    }
}


#endif
