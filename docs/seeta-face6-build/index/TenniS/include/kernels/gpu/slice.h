#ifndef TENSORSTACK_KERNELS_GPU_SLICE_H
#define TENSORSTACK_KERNELS_GPU_SLICE_H

#include "operator_on_gpu.h"
#include "backend/base/base_slice.h"


namespace ts {
    namespace gpu {
        class Slice : public OperatorOnGPU<base::Slice> {
        public:
            using self = Slice;
            using supper = OperatorOnGPU<base::Slice>;

            void slice(const Tensor &x, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_GPU_SLICE_H
