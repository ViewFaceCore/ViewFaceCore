//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H
#define TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H

#include <runtime/operator.h>
#include <runtime/stack.h>

namespace ts {
    class OperatorOnDevice : public Operator {
    public:
        virtual MemoryDevice running_memory_device() = 0;
    };

    /**
     * @tparam OP must be the sub class of Operator or OperatorOnDevice
     */
    template<typename OP>
    class OperatorOnAny : public OP {
    public:
        virtual MemoryDevice running_memory_device() {
            return this->memory_device();
        }
    };
}


#endif //TENSORSTACK_BACKEND_BASE_OPERATOR_ON_DEVICE_H
