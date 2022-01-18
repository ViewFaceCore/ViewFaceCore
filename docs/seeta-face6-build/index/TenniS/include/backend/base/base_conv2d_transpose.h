//
// Created by kier on 2019/2/21.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_CONV2D_TRANSPOSE_H
#define TENSORSTACK_BACKEND_BASE_BASE_CONV2D_TRANSPOSE_H

#include "operator_on_device.h"
#include <valarray>

#include "backend/common_structure.h"
#include "base_conv2d_transpose_core.h"

namespace ts {
    namespace base {
        class Conv2DTranspose : public OperatorOnDevice, public Conv2DTransposeCore {
        public:
            using self = Conv2DTranspose;
            using supper = OperatorOnDevice;

            Conv2DTranspose();

            void init() override;

            /**
             *
             * @param stack Contains x, w
             * @return 1
             * The w [input_channels, output_channels, height, width]
             * Notice the input_channels and output_channels swapped in weights, as it is the backward for Conv2D
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format;
            std::valarray<int> m_padding4x2;
            float m_padding_value;
            std::valarray<int> m_stride4;
            std::valarray<int> m_dilation4;

            std::valarray<int> m_output_shape;

            bool m_kernel_packed = false;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CONV2D_TRANSPOSE_H
