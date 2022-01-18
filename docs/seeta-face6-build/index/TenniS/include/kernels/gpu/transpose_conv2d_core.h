#ifndef TENSORSTACK_KERNELS_GPU_CONV2D_TRANSPOSE_CORE_H
#define TENSORSTACK_KERNELS_GPU_CONV2D_TRANSPOSE_CORE_H

#include "operator_on_gpu.h"
#include "backend/base/base_conv2d_transpose_core.h"


namespace ts {
    namespace gpu {
        class Conv2DTransposeCore : public base::Conv2DTransposeCore {
        public:
            using self = Conv2DTransposeCore;
            using supper = base::Conv2DTransposeCore;

            Conv2DTransposeCore() = default;

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack) override;

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                this->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }
        };
    }
}


#endif
