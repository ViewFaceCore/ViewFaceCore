//
// Created by kier on 2018/11/3.
//

#include <backend/name.h>
#include "runtime/instruction/tensor_instruction.h"

#include "runtime/workbench.h"

#include "runtime/instruction/instruction_factory.h"
#include "utils/static.h"
#include "core/tensor_builder.h"
#include "utils/assert.h"

namespace ts {
    namespace instruction {
        Instruction::shared Tensor::pack(size_t size) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench) {
                auto &stack = workbench.stack();
                if (stack.size() < size) {
                    TS_LOG(LOG_ERROR) << "Can not pack " << size << "tensor(s) on stack(size=" << stack.size() << ")"
                                      << eject;
                }
                std::vector<ts::Tensor> fields;
                fields.reserve(size);
                int anchor = -int(size);
                while (anchor < 0) {
                    fields.emplace_back(*stack.index(anchor));
                    ++anchor;
                }
                ts::Tensor packed_tensor;
                packed_tensor.pack(fields);
                stack.pop(size);
                stack.push(packed_tensor);
            }, "pack(" + std::to_string(size) + ")");
        }

        Instruction::shared Tensor::field(int index) {
            return std::make_shared<LambdaInstruction>([=](Workbench &workbench) {
                auto &stack = workbench.stack();
                auto field = stack.top()->field(index);
                stack.pop();
                stack.push(field);
            }, "field(" + std::to_string(index) + ")");
        }

        static std::vector<Instruction::shared> create_instruction_field(const Node &node) {
            TS_AUTO_CHECK(node.inputs().size() == 1);

            auto &bubble = node.bubble();
            TS_AUTO_CHECK(bubble.output_count() == 1);

            std::vector<Instruction::shared> inst;
            int offset = tensor::to_int(bubble.get(name::offset));

            TS_AUTO_CHECK(offset >= 0);

            inst.push_back(Tensor::field(offset));

            return inst;
        };

        static std::vector<Instruction::shared> create_instruction_pack(const Node &node) {
            auto &bubble = node.bubble();
            TS_AUTO_CHECK(bubble.output_count() == 1);

            std::vector<Instruction::shared> inst;

            inst.push_back(Tensor::pack(node.inputs().size()));

            return inst;
        };
    }
}

TS_STATIC_ACTION(ts::InstructionCreator::Register, ts::name::layer::field(), ts::instruction::create_instruction_field);
TS_STATIC_ACTION(ts::InstructionCreator::Register, ts::name::layer::pack(), ts::instruction::create_instruction_pack);
