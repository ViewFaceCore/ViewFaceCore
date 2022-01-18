//
// Created by kier on 2018/11/3.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H

#include "../instruction.h"

namespace ts {
    namespace instruction {
        class TS_DEBUG_API Tensor {
        public:
            // [-size, +1, e]
            static Instruction::shared pack(size_t size);

            // [-1, +1, e]
            static Instruction::shared field(int index);
        };
    }
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_TENSOR_INSTRUCTION_H
