#ifndef TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_CORE_H
#define TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_CORE_H

#include "backend/base/base_depthwise_conv2d_core.h"

namespace ts {
    namespace gpu {
        class DepthwiseConv2DCore : public base::DepthwiseConv2DCore {
        public:
            using self = DepthwiseConv2DCore;
            using supper = base::DepthwiseConv2DCore;

            DepthwiseConv2DCore() = default;

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_need_pack) override {
                if (!kernel_need_pack) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                conv2d(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_DEPTHWISE_CONV2D_CORE_H
