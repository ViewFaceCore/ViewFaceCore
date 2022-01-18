//
// Created by kier on 2019/9/7.
//

#ifndef TENSORSTACK_BACKEND_DRAGON_POOLING2D_AUTO_PAD_H
#define TENSORSTACK_BACKEND_DRAGON_POOLING2D_AUTO_PAD_H

#include "runtime/operator.h"
#include "backend/common_structure.h"

namespace ts {
    namespace dragon {
        class Pooling2DAutoPad : public Operator {
        public:
            using self = Pooling2DAutoPad;
            using supper = Operator;

            enum class AutoPadType {
                NOTSET,
                SAME_UPPER,
                SAME_LOWER,
                VALID,
            };

            Pooling2DAutoPad();

            void init() override;

            /**
             *
             * @param stack input_shape, ksize, stride
             * @return
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            AutoPadType auto_pad = AutoPadType::NOTSET;
            Padding2D static_padding;
            bool ceil_mode = true;
        };
    }
}


#endif //TENSORSTACK_BACKEND_DRAGON_POOLING2D_AUTO_PAD_H
