//
// Created by kier on 2018/11/3.
//

#include <runtime/instruction/instruction_factory.h>

#include "runtime/instruction/instruction_factory.h"

#include <map>

namespace ts {

    static std::map<std::string, InstructionCreator::function> &MapOpInstructionCreator() {
        static std::map<std::string, InstructionCreator::function> map_op_creator;
        return map_op_creator;
    };

    InstructionCreator::function InstructionCreator::Query(const std::string &op) TS_NOEXCEPT {
        auto &map_op_creator = MapOpInstructionCreator();
        auto op_creator = map_op_creator.find(op);
        if (op_creator != map_op_creator.end()) {
            return op_creator->second;
        }
        return function(nullptr);
    }

    void InstructionCreator::Register(const std::string &op,
                                      const InstructionCreator::function &builder) TS_NOEXCEPT {
        auto &map_op_creator = MapOpInstructionCreator();
        map_op_creator[op] = builder;
    }

    void InstructionCreator::Clear() {
        auto &map_op_creator = MapOpInstructionCreator();
        map_op_creator.clear();
    }
}
