#ifndef TENSORSTACK_KERNELS_CPU_ARGMAX_H
#define TENSORSTACK_KERNELS_CPU_ARGMAX_H

#include "operator_on_cpu.h"
#include "backend/base/base_argmax.h"


namespace ts {
    namespace cpu {
        class ArgMax : public OperatorOnCPU<base::ArgMax> {
        public:
            using self = ArgMax;
            using supper = OperatorOnCPU<base::ArgMax>;

            void argmax(const Tensor &x, int dim, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_ARGMAX_H
