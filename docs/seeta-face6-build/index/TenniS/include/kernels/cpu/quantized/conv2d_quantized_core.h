#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_QUANTIZED_CORE_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_QUANTIZED_CORE_H

#include "backend/base/base_conv2d_quantized_core.h"


namespace ts {
    namespace cpu {
        class Conv2DQuantizedCore : public base::Conv2DQuantizedCore {
        public:
            using self = Conv2DQuantizedCore;
            using supper = base::Conv2DQuantizedCore;

            Conv2DQuantizedCore() = default;

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, std::vector<float>dequantize_scales, 
                        Tensor &out, Stack &stack) override;
        };
    }
}


#endif //TENSORSTACK_KERNELS_CPU_CONV2D_QUANTIZED_CORE_H