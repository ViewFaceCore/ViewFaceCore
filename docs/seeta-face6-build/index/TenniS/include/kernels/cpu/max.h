#ifndef TENSORSTACK_KERNELS_CPU_MAX_H
#define TENSORSTACK_KERNELS_CPU_MAX_H

#include "operator_on_cpu.h"
#include "backend/base/base_max.h"


namespace ts {
    namespace cpu {
        class Max : public OperatorOnCPU<base::Max> {
        public:
            using self = Max;
            using supper = OperatorOnCPU<base::Max>;

            void max(const Tensor &x, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_MAX_H
