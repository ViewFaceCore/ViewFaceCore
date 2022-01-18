#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_CORE_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_CORE_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_core.h"


namespace ts {
    namespace cpu {
        class Conv2DCore : public base::Conv2DCore {
        public:
            using self = Conv2DCore;
            using supper = base::Conv2DCore;

            Conv2DCore() = default;

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override;
        };
    }
}


#endif //TENSORSTACK_KERNELS_CPU_CONV2D_CORE_H