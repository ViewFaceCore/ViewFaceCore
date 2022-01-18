//
// Created by kier on 2019/3/12.
//

#ifndef TENSORSTACK_BACKEND_ZOO_LIMIT_H
#define TENSORSTACK_BACKEND_ZOO_LIMIT_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class Limit : public Operator {
        public:
            using self = Limit;
            using supper = Operator;

            Limit();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::vector<int> m_shape;
            Operator::shared m_pad_op;
        };
    }
}


#endif //TENSORSTACK_BACKEND_ZOO_LIMIT_H
