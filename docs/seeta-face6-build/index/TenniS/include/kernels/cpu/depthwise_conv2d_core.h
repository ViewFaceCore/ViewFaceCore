#ifndef TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_CORE_H
#define TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_CORE_H

#include "backend/base/base_depthwise_conv2d_core.h"

namespace ts {
    namespace cpu {
        class DepthwiseConv2DCore : public base::DepthwiseConv2DCore {
        public:
            using self = DepthwiseConv2DCore;
            using supper = base::DepthwiseConv2DCore;

            DepthwiseConv2DCore() = default;

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_DEPTHWISE_CONV2D_CORE_H
