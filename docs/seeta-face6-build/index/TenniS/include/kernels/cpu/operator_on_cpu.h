//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_KERNELS_CPU_OPERATOR_ON_CPU_H
#define TENSORSTACK_KERNELS_CPU_OPERATOR_ON_CPU_H

#include "backend/base/operator_on_device.h"

namespace ts {
    namespace cpu {
        /**
         * @tparam OP must be the sub class of Operator or OperatorOnDevice
         */
        template<typename OP>
        class OperatorOnCPU : public OP {
        public:
            virtual MemoryDevice running_memory_device() {
                return MemoryDevice(CPU);
            }
        };
    }
}


#endif //TENSORSTACK_KERNELS_CPU_OPERATOR_ON_CPU_H
