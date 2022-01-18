//
// Created by kier on 2019/1/25.
//

#ifndef TENSORSTACK_BACKEND_DIVIDED_H
#define TENSORSTACK_BACKEND_DIVIDED_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class Divided : public Operator {
        public:
            using self = Divided;
            using supper = Operator;

            Divided();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::vector<int32_t> m_size;
            Operator::shared m_pad_op;

            Tensor m_padding;
        };
    }
}


#endif //TENSORSTACK_BACKEND_DIVIDED_H
