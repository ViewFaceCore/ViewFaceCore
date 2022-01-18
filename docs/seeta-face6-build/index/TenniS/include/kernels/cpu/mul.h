#ifndef TENSORSTACK_KERNELS_CPU_MUL_H
#define TENSORSTACK_KERNELS_CPU_MUL_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_cpu.h"
#include <backend/base/base_mul.h>

namespace ts {
    namespace cpu {
        class Mul : public OperatorOnCPU<base::Mul> {
        public:
            using self = Mul;
            using supper = OperatorOnCPU<base::Mul>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_MUL_H
