#ifndef TENSORSTACK_KERNELS_GPU_OPERATOR_ON_GPU_H
#define TENSORSTACK_KERNELS_GPU_OPERATOR_ON_GPU_H

#include "backend/base/operator_on_device.h"

#define CUDA_BLOCK(a,b)     (((a) + (b) - 1) / (b))
#define TRANS_BLOCK_DIM     16
#define CUDA_THREAD_NUM     512




namespace ts {
    namespace gpu {
        /**
         * @tparam OP must be the sub class of Operator or OperatorOnDevice
         */
        template<typename OP>
        class OperatorOnGPU : public OP {
        public:
            virtual MemoryDevice running_memory_device() {
                return MemoryDevice(GPU,this->memory_device().id());
            }
        };
    }
}


#endif //TENSORSTACK_KERNELS_GPU_OPERATOR_ON_GPU_H
