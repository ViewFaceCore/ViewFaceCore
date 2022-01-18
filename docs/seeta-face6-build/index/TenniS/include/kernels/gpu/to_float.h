#ifndef TENSORSTACK_KERNELS_GPU_TO_FLOAT_H
#define TENSORSTACK_KERNELS_GPU_TO_FLOAT_H


#include "kernels/gpu/operator_on_gpu.h"


namespace ts {
    namespace gpu {


        class ToFloat : public OperatorOnGPU<Operator> {
        public:
            using self = ToFloat;
            using supper = OperatorOnGPU<Operator>;

            ToFloat();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_castv2;
        };

    }
}


#endif //TENSORSTACK_KERNELS_GPU_TO_FLOAT_H

