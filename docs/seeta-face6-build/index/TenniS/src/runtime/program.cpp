//
// Created by kier on 2019-05-06.
//

#include <climits>
#include <compiler/argparse.h>

#include "runtime/program.h"
#include "compiler/compiler.h"
#include "core/tensor_builder.h"
#include "core/device_context.h"
#include "global/memory_device.h"

namespace ts {
    static std::string fuzzy_name(const Program::map<std::string, int> &map_name_slot, const std::string &name) {
        if (map_name_slot.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &name_slot_pair : map_name_slot) {
            auto &target_name = name_slot_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    template <typename T>
    inline void filter_values(T *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (near(*value, T(0))) *value = T(0);
            ++value;
        }
    }

    template <>
    inline void filter_values(float *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (*value < FLT_EPSILON && -*value < FLT_EPSILON) *value = 0;
            ++value;
        }
    }

    template <>
    inline void filter_values(double *value, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (*value < DBL_EPSILON && -*value < DBL_EPSILON) *value = 0;
            ++value;
        }
    }

    Program::shared Program::Compile(const Module::shared &module, const ComputingDevice &device, const std::string &options) {
        Program::shared program(new Program(device));
        // translate module
        auto translated_module = Module::Translate(module, device, options);
        // TODO: support RNN
        Compiler compiler(device);
        auto module_inputs = translated_module->inputs();
        auto module_outputs = translated_module->outputs();
        InstructionBlock block;

        // check workbench context
        {
            auto bench = ctx::of<Workbench>::get();
            if (bench == nullptr) {
                TS_LOG_ERROR << "Context<Workbench> need, but not bind." << eject;
            }
        }

        block = compiler.compile(module_inputs, module_outputs, options);

        // link data sagment
        // TODO: link multi-data-sagment
        auto data_sagment_base = int(program->m_data_segment->size());
        for (auto &inst : block.instructions) {
            auto data_sagment_inst = dynamic_cast<DataSegmentInstruction *>(inst.get());
            if (data_sagment_inst == nullptr) continue;
            inst = std::make_shared<DataSegmentInstruction>(data_sagment_inst->data_index() + data_sagment_base);
        }

        ArgParser parser;
        parser.add({"--filter", "-flt"}, {"--no-filter", "-no-flt"}, false);
        parser.parse(options);
        auto do_filter = parser.get("--filter");

        for (auto &data : block.data_segment) {
            Tensor *value = nullptr;
            if (data.device.empty()) {
                value = program->m_data_segment->clone_push(data.tensor);
            } else {
                value = program->m_data_segment->clone_push(data.tensor, data.device);
            }

            // filter value
            if (do_filter && value->device().type() == CPU) {
                if (value->dtype() == FLOAT32) {
                    filter_values(value->data<float>(), size_t(value->count()));
                } else if (value->dtype() == FLOAT64) {
                    filter_values(value->data<double>(), size_t(value->count()));
                }
            }
        }

        // binding instructions
        program->m_program = block.instructions;
        // binding input and output shots
        // program->m_inputs.resize(module_inputs.size());
        program->m_input_filters.resize(module_inputs.size());
        // program->m_outputs.resize(module_outputs.size());
        int slot_i = 0;
        for (auto &input : module_inputs) {
            program->m_map_input_slots.insert(std::make_pair(input.bubble().name(), slot_i++));
            program->m_input_names.emplace_back(input.bubble().name());
        }
        slot_i = 0;
        for (auto &output : module_outputs) {
            program->m_map_output_slots.insert(std::make_pair(output.bubble().name(), slot_i++));
            program->m_output_names.emplace_back(output.bubble().name());
        }

        program->m_input_dtypes.resize(module_inputs.size(), VOID);
        program->m_output_dtypes.resize(module_outputs.size(), VOID);

        for (size_t i = 0; i < module_inputs.size(); ++i) {
            auto &input = module_inputs[i];
            if (!input->has(Bubble::RetentionParam::dtype)) continue;
            program->m_input_dtypes[i] =
                    DTYPE(tensor::to_int(input->get(Bubble::RetentionParam::dtype)));
        }
        for (size_t i = 0; i < module_outputs.size(); ++i) {
            auto &output = module_outputs[i];
            if (!output->has(Bubble::RetentionParam::dtype)) continue;
            program->m_output_dtypes[i] =
                    DTYPE(tensor::to_int(output->get(Bubble::RetentionParam::dtype)));
        }

        return program;
    }

    Program::shared Program::clone() const {
        std::unique_lock<std::mutex> _lock_clone(*this->m_mutex);

        Program::shared dolly(new Program(this->m_device, this->m_mutex));
        dolly->m_program = this->m_program;
        // dolly->m_inputs.resize(this->m_inputs.size());
        // dolly->m_outputs.resize(this->m_outputs.size());
        dolly->m_map_input_slots = this->m_map_input_slots;
        dolly->m_map_output_slots = this->m_map_output_slots;
        dolly->m_data_segment = this->m_data_segment;
        dolly->m_input_filters.resize(this->m_input_filters.size(), nullptr);

        for (size_t i = 0; i < this->m_input_filters.size(); ++i) {
            if (this->m_input_filters[i] == nullptr) continue;
            dolly->m_input_filters[i] = this->m_input_filters[i]->clone();
        }

        // bind device context
        DeviceContext device_context(m_device);
        ctx::bind<DeviceContext> bind_device_context(device_context);

        for (auto &instruction : dolly->m_program) {
            auto op = dynamic_cast<OperatorInstruction*>(instruction.get());
            if (op == nullptr) continue;
            instruction = op->clone();
        }

        // copy dtype
        dolly->m_input_dtypes = m_input_dtypes;
        dolly->m_output_dtypes = m_output_dtypes;

        return std::move(dolly);
    }

    Program::Program(const ComputingDevice &device)
        : self(device, std::make_shared<std::mutex>()) {
    }

    Program::Program(const ComputingDevice &device, const std::shared_ptr<std::mutex> &mutex)
        : m_device(device), m_mutex(mutex) {
        auto memory_device = ComputingMemory::Query(m_device);

        this->m_data_segment = std::make_shared<Stack>(memory_device, DynamicSyncMemoryController::Make(memory_device, true));
    }

    Tensor Program::data_segment(int index) const {
        return this->m_data_segment->index(index)->weak();
    }

    Program::shared Program::input_filter(int slot) const {
        if (slot < 0 || slot >= input_count()) {
            TS_LOG_ERROR << "Input index out of range[0, " << input_count() << "). with index=" << slot << eject;
        }
        return this->m_input_filters[slot];
    }

    DTYPE Program::input_dtype(int slot) const {
        if (slot < 0 || slot >= input_count()) {
            TS_LOG_ERROR << "Input index out of range[0, " << input_count() << "). with index=" << slot << eject;
        }
        return this->m_input_dtypes[slot];
    }

    int Program::input_slot(const std::string &name) const {
        auto slot_it = m_map_input_slots.find(name);
        if (slot_it == m_map_input_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_input_slots, name) << eject;
        }
        return slot_it->second;
    }

    int Program::output_slot(const std::string &name) const {
        auto slot_it = m_map_output_slots.find(name);
        if (slot_it == m_map_output_slots.end()) {
            TS_LOG_ERROR << "Can not identify the name \"" << name << "\", did you mean: "
                         << fuzzy_name(m_map_output_slots, name) << eject;
        }
        return slot_it->second;
    }

    int Program::input_count() const {
        return int(m_input_dtypes.size());
    }

    int Program::output_count() const {
        return int(m_output_dtypes.size());
    }

    size_t Program::length() const {
        return m_program.size();
    }

    const Instruction::shared &Program::instruction(size_t pointer) const {
        return m_program[pointer];
    }

    const std::vector<Instruction::shared> &Program::instruction() const {
        return m_program;
    }

    void Program::bind_filter(int slot, Program::shared filter) {
        if (slot < 0 || slot >= input_count()) {
            TS_LOG_ERROR << "Input index out of range[0, " << input_count() << "). with index=" << slot << eject;
        }
        if (filter != nullptr) {
            if (filter->input_count() != 1 || filter->output_count() != 1) {
                TS_LOG_ERROR << "Filter's input count and output count must both be 1." << eject;
            }
        }
        this->m_input_filters[slot] = filter;
    }

    void Program::set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value) {
        for (auto &inst : this->instruction()) {
            auto *operator_inst = dynamic_cast<OperatorInstruction *>(inst.get());
            if (operator_inst == nullptr) continue;
            auto node = operator_inst->op();
            if (node->name() != node_name) continue;
            node->set(param, value);
            node->init();
        }
    }

    Program::shared Program::Compile(const Module::shared &module, const ComputingDevice &device) {
        return Compile(module, device, "");
    }

    const Stack &Program::data_segment() const {
        return *m_data_segment;
    }

    const std::vector<std::string> &Program::input_names() const {
        return m_input_names;
    }

    const std::vector<std::string> &Program::output_names() const {
        return m_output_names;
    }
}