//
// Created by kier on 2018/10/17.
//

#ifndef TENSORSTACK_COMPILER_COMPILER_H
#define TENSORSTACK_COMPILER_COMPILER_H

#include "runtime/instruction.h"
#include "module/graph.h"

namespace ts {

    class DeviceTensor
    {
    public:
        Tensor tensor;
        DeviceType device;

        DeviceTensor(const Tensor &tensor) : tensor(tensor) {}
        DeviceTensor(const Tensor &tensor, const DeviceType &device)
            : tensor(tensor), device(device) {}

        operator Tensor() const { return tensor; }
    };

    class InstructionBlock
    {
    public:
        int nargs = 0;
        int nresults = 0;
        std::vector<Instruction::shared> instructions;
        std::vector<DeviceTensor> data_segment;
    };

    /**
     * compile the ZGraph to instructions
     */
    class Compiler {
    public:
        using self = Compiler;

        explicit Compiler(const ComputingDevice &computing_device);

        InstructionBlock compile(const std::vector<Node> &raw_inputs, const std::vector<Node> &outputs);
        InstructionBlock compile(const std::vector<Node> &raw_inputs, const std::vector<Node> &outputs,
                const std::string &options);
        std::vector<Instruction::shared> convert_operator_instruction(const Node &node);

        static void run_const_nodes(const std::vector<Node> &nodes, std::vector<Node> &const_nodes);
    private:
        ComputingDevice m_computing_device;
    };
}


#endif //TENSORSTACK_COMPILER_COMPILER_H
