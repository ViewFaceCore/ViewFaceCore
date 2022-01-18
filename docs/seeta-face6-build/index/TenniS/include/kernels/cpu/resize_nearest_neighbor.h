#ifndef TENSORSTACK_KERNELS_CPU_RESIZE_NEAREST_NEIGHBOR_H
#define TENSORSTACK_KERNELS_CPU_RESIZE_NEAREST_NEIGHBOR_H

#include "operator_on_cpu.h"


namespace ts {
    namespace cpu {
        class Resize_Nearest_Neighbor : public OperatorOnAny<Operator> {
        public:
            using self = Resize_Nearest_Neighbor;
            using supper = OperatorOnAny<Operator>;

            Resize_Nearest_Neighbor();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_resize2d;
            int m_align_corners;
            int m_dim;
        };
    }
}


#endif
