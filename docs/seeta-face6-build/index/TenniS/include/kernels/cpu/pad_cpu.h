#ifndef TENSORSTACK_KERNELS_CPU_PAD_H
#define TENSORSTACK_KERNELS_CPU_PAD_H

#include "operator_on_cpu.h"
#include "backend/base/base_pad.h"


namespace ts {
    namespace cpu {
        class PadOnCPU : public OperatorOnCPU<base::Pad> {
        public:
            using self = Pad;
            using supper = OperatorOnCPU<base::Pad>;

            void pad(const Tensor &x, const std::vector<std::array<int, 2>> &padding, float padding_value, Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_PAD_H
