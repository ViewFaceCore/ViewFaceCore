#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_V2_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_V2_H

#include "operator_on_cpu.h"
#include "backend/base/base_conv2d_v2.h"
#include "conv2d_core.h"


namespace ts {
    namespace cpu {
        // using Conv2DV2 = base::Conv2DWithCore<OperatorOnCPU<base::Conv2DV2>, Conv2DCore>;
        class Conv2DV2 : public OperatorOnAny<Operator> {
        public:
            using self = Conv2DV2;
            using supper = OperatorOnAny<Operator>;

            Conv2DV2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_conv2d;
            Tensor m_int_padding4x2;    // save pre set padding
        };
    }
}


#endif //TENSORSTACK_KERNELS_CPU_CONV2D_V2_H