#ifndef TENSORSTACK_KERNELS_CPU_TRANSPOSE_H
#define TENSORSTACK_KERNELS_CPU_TRANSPOSE_H

#include "backend/base/base_transpose.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class Transpose : public OperatorOnCPU<base::Transpose> {
        public:
            using self = Transpose;
            using supper = OperatorOnCPU<base::Transpose>;

            void transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_TRANSPOSE_H
