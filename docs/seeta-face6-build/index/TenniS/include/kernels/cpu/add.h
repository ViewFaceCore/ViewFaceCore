#ifndef TENSORSTACK_KERNELS_CPU_ADD_H
#define TENSORSTACK_KERNELS_CPU_ADD_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_cpu.h"
#include <backend/base/base_add.h>

namespace ts {
    namespace cpu {
        class Add : public OperatorOnCPU<base::Add> {
        public:
            using self = Add;
            using supper = OperatorOnCPU<base::Add>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

        };
    }
}

#endif
