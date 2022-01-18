#ifndef TENSORSTACK_KERNELS_CPU_PAD_H
#define TENSORSTACK_KERNELS_CPU_PAD_H

#include "kernels/gpu/operator_on_gpu.h"
#include "backend/base/base_pad.h"


namespace ts {
    namespace gpu {
        class PadOnGPU : public OperatorOnGPU<base::Pad> {
        public:
            using self = Pad;
            using supper = OperatorOnGPU<base::Pad>;

            void pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_PAD_H
