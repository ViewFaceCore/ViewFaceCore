//
// Created by kier on 2019/3/17.
//

#ifndef TENSORSTACK_BACKEND_TF_CONV2D_AUTO_PAD_H
#define TENSORSTACK_BACKEND_TF_CONV2D_AUTO_PAD_H

#include "runtime/operator.h"
#include "backend/common_structure.h"

namespace ts {
    namespace tf {
        class Conv2DPadding : public Operator {
        public:
            using self = Conv2DPadding;
            using supper = Operator;

            enum class PaddingMethod {
                SAME,
                VALID,
            };

            Conv2DPadding();

            void init() override;

            /**
             *
             * @param stack x, weight
             * @return
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            PaddingMethod m_padding_method = PaddingMethod::SAME;
            Padding2D m_static_padding;
            Conv2DFormat m_format;
            Stride2D m_stride;
            Dilation2D m_dilation;
        };
    }
}


#endif //TENSORSTACK_BACKEND_TF_CONV2D_AUTO_PAD_H
