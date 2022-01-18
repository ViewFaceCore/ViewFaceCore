//
// Created by kier on 2019/1/19.
//

#ifndef TENSORSTACK_BACKEND_ZOO_COPY_H
#define TENSORSTACK_BACKEND_ZOO_COPY_H


#include <runtime/operator.h>

namespace ts {
    namespace zoo {
        class Copy : public Operator {
        public:
            using self = Copy;
            using supper = Operator;

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            int m_output_count = 1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_ZOO_COPY_H
