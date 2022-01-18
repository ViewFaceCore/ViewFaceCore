//
// Created by kier on 2019/3/17.
//

#ifndef TENSORSTACK_BACKEND_TF_POOLING2D_AUTO_PAD_H
#define TENSORSTACK_BACKEND_TF_POOLING2D_AUTO_PAD_H

#include "runtime/operator.h"
#include "backend/common_structure.h"

namespace ts {
    namespace tf {
        class Pooling2DPadding : public Operator {
        public:
            using self = Pooling2DPadding;
            using supper = Operator;

            enum class PaddingMethod {
                SAME,
                VALID,
            };

            Pooling2DPadding();

            void init() override;

            /**
             *
             * @param stack input_shape, ksize, stride
             * @return
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format = Conv2DFormat::FORMAT_NCHW;
            PaddingMethod m_padding_method = PaddingMethod::SAME;
            Padding2D m_static_padding;
        };
    }
}


#endif //TENSORSTACK_BACKEND_TF_POOLING2D_AUTO_PAD_H
