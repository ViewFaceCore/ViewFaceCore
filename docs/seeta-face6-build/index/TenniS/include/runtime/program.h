//
// Created by kier on 2019-05-06.
//

#ifndef TENSORSTACK_RUNTIME_PROGRAM_H
#define TENSORSTACK_RUNTIME_PROGRAM_H


#include <memory>
#include "module/module.h"
#include "runtime/stack.h"
#include "runtime/instruction.h"

namespace ts {
    class TS_DEBUG_API Program {
    public:
        using self = Program;
        using shared = std::shared_ptr<self>;

        template<typename K, typename V>
        using map = std::unordered_map<K, V>;

        /**
         *
         * @param module
         * @param device
         * @return
         * @context Workbench for compile usage
         */
        static shared Compile(const Module::shared &module, const ComputingDevice &device);

        /**
         *
         * @param module
         * @param device
         * @param options like --winograd
         * @return
         * @context Workbench for compile usage
         */
        static shared Compile(const Module::shared &module, const ComputingDevice &device, const std::string &options);

        shared clone() const;

        void bind_filter(int slot, shared filter);

        shared input_filter(int slot) const;

        DTYPE input_dtype(int slot) const;

        int input_slot(const std::string &name) const;

        int output_slot(const std::string &name) const;

        int input_count() const;

        int output_count() const;

        const ComputingDevice &device() const {  return m_device; }

        Tensor data_segment(int index) const;

        size_t length() const;

        const Instruction::shared &instruction(size_t pointer) const;

        const std::vector<Instruction::shared> &instruction() const;

        void set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value);

        const Stack &data_segment() const;

        const std::vector<std::string> &input_names() const;

        const std::vector<std::string> &output_names() const;

    private:
        Program(const ComputingDevice &device);
        Program(const ComputingDevice &device, const std::shared_ptr<std::mutex> &mutex);

        ComputingDevice m_device;

        std::vector<Instruction::shared> m_program; // running function, program area

        Stack::shared m_data_segment;   // save static area
        // map slot, means <tensor'name, tensor's index in stack>
        map<std::string, int> m_map_input_slots;
        map<std::string, int> m_map_output_slots;
        // map tensor, means <tensor's index in stack, tensor>
        // std::vector<Tensor> m_inputs;
        // std::vector<Tensor> m_outputs;
        // input and output dtype type
        std::vector<DTYPE> m_input_dtypes;
        std::vector<DTYPE> m_output_dtypes;

        std::vector<shared> m_input_filters;    // program filter

        std::shared_ptr<std::mutex> m_mutex;

        std::vector<std::string> m_input_names;
        std::vector<std::string> m_output_names;
    };

    class TS_DEBUG_API ProgramEnv {
    public:
        using self = ProgramEnv;
        using shared = std::shared_ptr<self>;

        ProgramEnv(Program::shared program)
                : program(std::move(program)) {
            this->length = this->program->length();
        }

        Program::shared program;
        size_t pointer = 0;
        size_t length = 0;
    };
}


#endif //TENSORSTACK_RUNTIME_PROGRAM_H
