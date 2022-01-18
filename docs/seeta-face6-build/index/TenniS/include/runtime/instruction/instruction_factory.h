//
// Created by kier on 2018/11/3.
//

#ifndef TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H
#define TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H

#include <functional>

#include "../instruction.h"
#include "module/graph.h"

namespace ts {
    // TODO: add instruction factory, query instruction by name
    // Those instructions are cross computing and memory device operator
    class TS_DEBUG_API InstructionCreator {
    public:
        using function = std::function<std::vector<Instruction::shared>(const Node &)>;

        /**
         * Example of InstructionBuilder
         * @param node node ready to convert to instruction
         * @return an serial of instructions, those can calculate node
         */
        std::vector<Instruction::shared> InstructionCreatorrFunction(const Node &node);

        /**
         * Query instruction builder of specific op
         * @param op querying op
         * @return InstructionBuilder
         * @note supporting called by threads without calling @sa RegisterInstructionBuilder
         * @note the query should be the Bubble.op
         */
        static function Query(const std::string &op) TS_NOEXCEPT;

        /**
         * Register InstructionBuilder for specific op
         * @param op specific op name @sa Bubble
         * @param builder instruction builder
         * @note only can be called before running @sa QueryInstructionBuilder
         */
        static void Register(const std::string &op, const function &builder) TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();
    };
}


#endif //TENSORSTACK_RUNTIME_INSTRUCTION_INSTRUCTION_FACTORY_H
