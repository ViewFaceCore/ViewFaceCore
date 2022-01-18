#ifndef TENSORSTACK_KERNELS_GPU_MAX_H
#define TENSORSTACK_KERNELS_GPU_MAX_H

#include "operator_on_gpu.h"
#include "backend/base/base_max.h"


namespace ts {
    namespace gpu {
        class Max : public OperatorOnGPU<base::Max> {
        public:
            using self = Max;
            using supper = OperatorOnGPU<base::Max>;

            void max(const Tensor &x, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_MAX_H
