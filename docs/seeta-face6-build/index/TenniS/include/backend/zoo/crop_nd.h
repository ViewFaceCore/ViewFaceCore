//
// Created by kier on 2019/4/9.
//

#ifndef TENSORSTACK_BACKEND_NHWC_CROPND_H
#define TENSORSTACK_BACKEND_NHWC_CROPND_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class CropND : public Operator {
        public:
            using self = CropND;
            using supper = Operator;

            CropND();

            void init() override;

            /**
             *
             * @param stack x, size
             * @return cropped tensor
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::vector<int> m_shift;
            Operator::shared m_pad_op;
        };
    }
}


#endif //TENSORSTACK_BACKEND_NHWC_CROPND_H
