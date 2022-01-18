#ifndef TENSORSTACK_KERNELS_CPU_RANGE_H
#define TENSORSTACK_KERNELS_CPU_RANGE_H

#include "operator_on_cpu.h"


namespace ts {
    namespace cpu {
        class Range : public OperatorOnAny<Operator> {
        public:
            using self = Range;
            using supper = OperatorOnAny<Operator>;

            Range();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        };
    }
}


#endif  // TENSORSTACK_KERNELS_CPU_RANGE_H
