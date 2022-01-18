//
// Created by kier on 2018/10/17.
//

#include "runtime/instruction/stack_instruction.h"

#include "runtime/workbench.h"

namespace ts {
    namespace instruction {
        Instruction::shared Stack::push(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().push(i);
            }, "push(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::clone(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().clone(i);
            }, "clone(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::erase(int i) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().erase(i);
            }, "erase(" + std::to_string(i) + ")");
        }

        Instruction::shared Stack::ring_shift_left() {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                stack.push(0);
                stack.erase(0);
            }, "<<<(" + std::to_string(1) + ")");
        }

        Instruction::shared Stack::swap(int i, int j) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                auto &stack = workbench.stack();
                auto ti = *stack.index(i);
                auto tj = *stack.index(j);
                *stack.index(i) = tj;
                *stack.index(j) = ti;
            }, "swap(" + std::to_string(i) + ", " + std::to_string(j) + ")");
        }

        Instruction::shared Stack::erase(int beg, int end) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench){
                workbench.stack().erase(beg, end);
            }, "erase(" + std::to_string(beg) + ", " + std::to_string(end) + ")");
        }
    }
}

