#ifndef TENSORSTACK_KERNELS_GPU_ARGMAX_H
#define TENSORSTACK_KERNELS_GPU_ARGMAX_H

#include "operator_on_gpu.h"
#include "backend/base/base_argmax.h"


namespace ts {
    namespace gpu {
        class ArgMax : public OperatorOnGPU<base::ArgMax> {
        public:
            using self = ArgMax;
            using supper = OperatorOnGPU<base::ArgMax>;

            void argmax(const Tensor &x, int dim, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_ARGMAX_H
