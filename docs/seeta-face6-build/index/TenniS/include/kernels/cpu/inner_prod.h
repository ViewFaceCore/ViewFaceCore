#ifndef TENSORSTACK_KERNELS_CPU_INNER_PROD_H
#define TENSORSTACK_KERNELS_CPU_INNER_PROD_H

#include "operator_on_cpu.h"
#include "backend/base/base_inner_prod.h"


namespace ts {
    namespace cpu {
        class InnerProd : public OperatorOnCPU<base::InnerProd> {
        public:
            using self = InnerProd;
            using supper = OperatorOnCPU<base::InnerProd>;

            void inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out, Stack &stack, bool kernel_packed) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_INNER_PROD_H
