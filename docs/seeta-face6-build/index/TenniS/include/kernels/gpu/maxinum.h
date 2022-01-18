#ifndef TENSORSTACK_KERNELS_GPU_MAXINUM_H
#define TENSORSTACK_KERNELS_GPU_MAXINUM_H

#include <core/tensor.h>
#include <runtime/stack.h>
#include "operator_on_gpu.h"
#include <backend/base/element_wise_reduce.h>

namespace ts {
    namespace gpu {
        class Maxinum : public OperatorOnGPU<ElementWiseReduce> {
        public:
            using self = Maxinum;
            using supper = OperatorOnGPU<ElementWiseReduce>;

            void reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

            void reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) override;

            void reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) override;

        };
    }
}

#endif
