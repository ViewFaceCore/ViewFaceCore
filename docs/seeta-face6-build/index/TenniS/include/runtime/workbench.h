//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_RUNTIME_WORKBENCH_H
#define TENSORSTACK_RUNTIME_WORKBENCH_H

#include <memory>
#include <queue>
#include <unordered_map>
#include <stack>

#include <core/device_context.h>

#include "operator.h"
#include "stack.h"
#include "core/tensor.h"
#include "instruction.h"
#include "module/module.h"
#include "inside/thread_pool.h"
#include "global/device_admin.h"
#include "runtime/runtime.h"
#include "image_filter.h"
#include "board/profiler.h"

#include "utils/ctxmgr_lite.h"
#include "utils/cpu.h"

#include "program.h"
#include "runtime/switcher.h"

namespace ts {
    class TS_DEBUG_API Workbench : public SetupContext<Workbench> {
    public:
        using self = Workbench;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        template<typename K, typename V>
        using map = std::unordered_map<K, V>;

        explicit Workbench(const ComputingDevice &device);

        explicit Workbench(const ComputingDevice &device, int computing_thread_number);

        explicit Workbench(const ComputingDevice &device, CpuEnable::CpuPowerMode cpu_mode);

        ~Workbench();

        Workbench(const self &) = delete;

        Workbench &operator=(const self &) = delete;

        Stack &stack() { return *this->m_stack; }

        const Stack &stack() const { return *this->m_stack; }

        void jump_relative(int shift);

        void jump_absolute(size_t pointer);

        void push_data_segment(int data_index);

        // clear all stack
        void clear() { this->m_stack->clear(); }

        // bind image filter
        void bind_filter(const std::string &name, ImageFilter::shared filter);

        void bind_filter(int slot, ImageFilter::shared filter);

        // set input
        void input(const std::string &name, const Tensor &tensor);

        void input(int slot, const Tensor &tensor);

        const Tensor &input(const std::string &name) const;

        const Tensor &input(int slot) const;

        // run graph
        void run();

        // get output
        const Tensor &output(const std::string &name) const;

        const Tensor &output(int slot) const;

        int input_count() const;

        int output_count() const;

        // clone an Workbench which can run
        Workbench::shared clone() const;

        static shared Load(const Module::shared &module, const ComputingDevice &device);

        static shared Load(const Module::shared &module, const ComputingDevice &device, const std::string &options);

        const DeviceContext &device() const { return m_device_context; }

        DeviceContext &device() { return m_device_context; }

        const RuntimeContext &runtime() const { return m_runtime_context; }

        RuntimeContext &runtime() { return m_runtime_context; }

        void do_profile(bool _do) { m_do_profile = _do; }

        Profiler &profiler() { return m_profiler; }

        const Profiler &profiler() const { return m_profiler; }

        /**
         *
         * @param [in] bubble parameter
         * @param [in] device operator device
         * @param [in] strict if it in strict mode
         * @return operator
         */
        Operator::shared offline_create(const Bubble &bubble, bool strict = true);

        /**
         *
         * @param [in] op testing operator
         * @param [in] input input tensor
         * @param [out] output output tensor
         */
        void offline_run(Operator::shared op, const std::vector<Tensor> &input, std::vector<Tensor> &output);

        /**
         *
         * @param [in] op testing operator
         * @param [in] input input operators
         * @param [out] output output oprators
         */
        void offline_infer(Operator::shared op, const std::vector<Tensor> &input, std::vector<Tensor::Prototype> &output);

        /**
         *
         * @param [in] bubble parameter
         * @param [in] device operator device
         * @param [in] strict if it in strict mode
         * @return operator
         */
        Operator::shared online_create(const Bubble &bubble, bool strict = false);

        /**
         * it wont change stack before op
         * @param op
         */
        int online_run(Operator::shared op, int argc);

        /**
         * This API push input to stack, then run online
         * @param op
         * @param input
         */
        int online_run(Operator::shared op, const std::vector<Tensor> &input);

        /**
         * it wont change stack before inst
         * @param inst
         */
        void online_run(Instruction::shared inst);

        /**
         * This API push input to stack, then run online, the inst return size, is given by doc
         * @param inst
         * @param input
         */
        void online_run(Instruction::shared inst, const std::vector<Tensor> &input);

        void set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value);

        /**
         * it wont change stack before op
         * @param op
         */
        int online_run(const Bubble &bubble, int argc, bool strict = false);

        /**
         * This API will clear stack before run op, then push input to stack
         * @param op
         * @param input
         */
        int online_run(const Bubble &bubble, const std::vector<Tensor> &input, bool strict = false);

        /**
         *
         * @param program
         * This function will reset all programs. make older run api working
         */
        void setup(Program::shared program);

        /**
         *
         * @param program
         * @param nargs
         * @return output count in stack
         * This function will keep ready stack, run with args on stack
         */
        int launch_online(Program::shared program, int nargs);

        /**
         *
         * @param program
         * @param args
         * @return
         * This function will keep ready stack, there will be nothing happen after launch,
         * call launch_online inner
         */
        std::vector<Tensor> launch_offline(Program::shared program, const std::vector<Tensor> &args);

        /**
         *
         * @param program
         * @param args
         * @return
         * This function will keep ready stack, there will be nothing happen after launch
         * call other launch_offline inner
         */
        std::vector<Tensor> launch_offline(Program::shared program, const std::map<std::string, Tensor> &args);

        Program::shared compile(const Module::shared &module);

        Program::shared compile(const Module::shared &module, const std::string &options);

        /**
         * setup context DeviceContext
         */
        void setup_device();

        /**
         * setup context RuntimeContext
         */
        void setup_runtime();

        /**
         *
         */
        void run_hook(const std::vector<std::string> &node_names);

        const std::string &summary();

        /**
        *
        * @param [in] cpu_mode cpu power mode parameter
        * @return return true is success
        * only support android system now
        */
        bool set_cpu_power_mode(CpuEnable::CpuPowerMode cpu_mode);

        SwitchControll::shared switch_controller();

    private:
        // size_t m_pointer = 0;   // pointer to running function
        // std::vector<Instruction::shared> m_program; // running function, program area
        SyncMemoryController::shared m_static_memory;
        SyncMemoryController::shared m_flow_memory;
        SyncMemoryController::shared m_dynamic_memory;
        Stack::shared m_stack;  // save running memory, data area
        // Stack::shared m_data_sagment;   // save static area
        // map slot, means <tensor'name, tensor's index in stack>
        // map<std::string, int> m_map_input_slots;
        // map<std::string, int> m_map_output_slots;
        // map tensor, means <tensor's index in stack, tensor>
        std::vector<Tensor> m_inputs;
        std::vector<Tensor> m_outputs;
        // input and output dtype type
        // std::vector<DTYPE> m_input_dtypes;
        // std::vector<DTYPE> m_output_dtypes;

        // std::vector<ImageFilter::shared> m_input_filters;

        // control device context
        DeviceContext m_device_context;

        // runtime setting, shared in working thread
        RuntimeContext m_runtime_context;

        bool m_do_profile = false;
        Profiler m_profiler;

        // std::shared_ptr<std::mutex> m_mutex;

        std::stack<ProgramEnv> m_env;

        Program::shared m_desktop;

        std::map<std::string, Tensor> m_hooked_tensor;

        std::string m_summary;

        SwitchControll::shared m_switch_controller;
    private:
        Operator::shared m_cast_op; ///< for input cast

        void cast_tensor(DTYPE dtype);
    };
}


#endif //TENSORSTACK_RUNTIME_WORKBENCH_H
