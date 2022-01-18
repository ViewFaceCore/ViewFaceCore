//
// Created by kier on 2019/1/12.
//

#ifndef TENSORSTACK_BACKEND_MXNET_POOLING2D_PADDING_H
#define TENSORSTACK_BACKEND_MXNET_POOLING2D_PADDING_H

#include "runtime/operator.h"
#include "backend/common_structure.h"

namespace ts {
    namespace mxnet {
        class Pooling2dPadding : public Operator {
        public:
            using self = Pooling2dPadding;
            using supper = Operator;

            Pooling2dPadding();

            void init() override;

            /**
             *
             * @param stack input_shape, ksize, stride
             * @return
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::string format;
            Padding2D static_padding;
            bool valid;
        };
    }
}


#endif //TENSORSTACK_BACKEND_MXNET_POOLING2D_PADDING_H
