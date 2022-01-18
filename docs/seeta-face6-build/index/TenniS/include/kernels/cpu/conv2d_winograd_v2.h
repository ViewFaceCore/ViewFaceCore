//
// Created by yang on 2019/11/8.
//

#ifndef TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_V2_H
#define TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_V2_H

#include "operator_on_cpu.h"

namespace ts{
    namespace cpu{
        class Conv2DWinogradV2 : public OperatorOnAny<Operator>{
        public:
            using self = Conv2DWinogradV2;
            using supper = OperatorOnAny<Operator>;

            Conv2DWinogradV2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Operator::shared m_op_conv2d_winograd;
            Tensor m_int_padding4x2; // save pre set padding
        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_CONV2D_WINOGRAD_V2_H
