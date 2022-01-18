#ifndef TENSORSTACK_KERNELS_CPU_QUANTIZE_H
#define TENSORSTACK_KERNELS_CPU_QUANTIZE_H

#include "backend/base/base_quantize.h"
#include "operator_on_cpu.h"

namespace ts {
    namespace cpu {
        class Quantize : public OperatorOnCPU<base::Quantize> {
        public:
            using self = Quantize;
            using supper = ts::Operator;

            void quantize(const Tensor &x, std::vector<float> quantize_scales, Tensor &out) override;
        };
    }
}



#endif //TENSORSTACK_KERNELS_CPU_PRELU_H