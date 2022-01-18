#ifndef TENSORSTACK_KERNELS_GPU_TRANSPOSE_H
#define TENSORSTACK_KERNELS_GPU_TRANSPOSE_H

#include "backend/base/base_transpose.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class Transpose : public OperatorOnGPU<base::Transpose> {
        public:
            using self = Transpose;
            using supper = OperatorOnGPU<base::Transpose>;

            void transpose(const Tensor &x, const std::vector<int> &permute, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_TRANSPOSE_H
