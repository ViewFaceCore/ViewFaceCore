#ifndef TENSORSTACK_BACKEND_BASE_BASE_CONV2D_QUANTIZED_H
#define TENSORSTACK_BACKEND_BASE_BASE_CONV2D_QUANTIZED_H

#include "operator_on_device.h"
#include <valarray>

#include "backend/common_structure.h"
#include "base_conv2d_quantized_core.h"

namespace ts {
    namespace base {
        class Conv2DQuantized : public OperatorOnDevice, public Conv2DQuantizedCore {
        public:
            using self = Conv2DQuantized;
            using supper = OperatorOnDevice;

            Conv2DQuantized();

            void init() override;

            /**
            *
            * @param stack Contains x, w
            * @return 1
            */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format;
            std::valarray<int> m_padding4x2;
            float m_padding_value;
            std::valarray<int> m_stride4;
            std::valarray<int> m_dilation4;

            //float m_quantize_scale;
            std::vector<float> m_dequantize_scales;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CONV2D_QUANTIZED_H
