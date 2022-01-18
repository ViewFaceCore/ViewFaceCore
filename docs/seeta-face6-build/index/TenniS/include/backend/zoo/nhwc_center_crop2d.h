//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_BACKEND_NHWC_CENTER_CROP2D_H
#define TENSORSTACK_BACKEND_NHWC_CENTER_CROP2D_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class NHWCCenterCrop2D : public Operator {
        public:
            using self = NHWCCenterCrop2D;
            using supper = Operator;

            NHWCCenterCrop2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Size2D m_size;
            Operator::shared m_pad_op;
        };
    }
}


#endif //TENSORSTACK_BACKEND_NHWC_CENTER_CROP2D_H
