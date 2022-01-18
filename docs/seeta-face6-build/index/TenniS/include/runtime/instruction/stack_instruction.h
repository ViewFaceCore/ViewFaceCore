//
// Created by kier on 2018/10/17.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H

#include "../instruction.h"

namespace ts {
    namespace instruction {
        class TS_DEBUG_API Stack {
        public:
            // [-0, +1, -]
            static Instruction::shared push(int i);
            // [-0, +1, -]
            static Instruction::shared clone(int i);
            // [-1, +0, -]
            static Instruction::shared erase(int i);
            // [-(end - beg), +0, -]
            static Instruction::shared erase(int beg, int end);
            // [-0, +0, -]
            static Instruction::shared ring_shift_left();
            // [-0, +0, -]
            static Instruction::shared swap(int i, int j);
        };
    }
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_ISTACK_H
