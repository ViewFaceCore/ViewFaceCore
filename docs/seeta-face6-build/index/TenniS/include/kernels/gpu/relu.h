#ifndef TENSORSTACK_KERNELS_GPU_RELU_H
#define TENSORSTACK_KERNELS_GPU_RELU_H

#include "backend/base/base_relu.h"
#include "operator_on_gpu.h"

namespace ts {
    namespace gpu {
        class ReLU : public OperatorOnGPU<base::ReLU> {
        public:
            using self = ReLU;
            using supper = ts::Operator;

            void active(const Tensor &x, Tensor &out) override;
        };
    }
}



#endif //TENSORSTACK_KERNELS_GPU_RELU_H