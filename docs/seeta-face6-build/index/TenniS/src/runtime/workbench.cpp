//
// Created by kier on 2018/5/25.
//

#include <module/module.h>
#include <core/device.h>
#include <runtime/workbench.h>
#include <utils/ctxmgr.h>

#include "runtime/workbench.h"
#include "global/memory_device.h"
#include "compiler/compiler.h"
#include "utils/box.h"

#include "memory/flow.h"
#include "global/hard_converter.h"

#include <climits>
#include <board/hook.h>
#include "utils/need.h"

#include "kernels/common/math.h"
#include "core/tensor_builder.h"
#include "backend/name.h"
#include "backend/base/base_cast_v2.h"
#include "runtime/operator.h"

#include "utils/ctxmgr_lite_support.h"
#include "utils/cpu_info.h"

namespace ts {
    class BindWorkbenchRuntime {
    public:
        using self = BindWorkbenchRuntime;

        explicit BindWorkbenchRuntime(Workbench &bench)
            : bind_thread_pool(bench.runtime().thread_pool())
            // , bind_device_context(bench.device())
            , bind_runtime_context(bench.runtime())
            , bind_work_bench(bench) {
            // bench.device().active();
            m_pre_device_context = DeviceContext::Switch(&bench.device());

            auto switch_controller = bench.switch_controller();
            if(switch_controller->is_load_dll()){
                switch_controller->bind_context();
            }
        }

        ~BindWorkbenchRuntime() {
            DeviceContext::Switch(m_pre_device_context);
        }

    private:
        // bind thread pool to any operator can using thread speed up
        ctx::bind<ThreadPool> bind_thread_pool;

        // bind device context
        // ctx::bind<DeviceContext> bind_device_context;

        // bind runtime context
        ctx::bind<RuntimeContext> bind_runtime_context;

        // pre_device_context
        DeviceContext *m_pre_device_context = nullptr;

        // bind self
        ctx::bind<Workbench> bind_work_bench;
    };

    static std::string feature_log(const std::vector<CPUFeature> &features) {
        std::ostringstream oss;
        oss << "{";
        for (size_t i = 0; i < features.size(); ++i) {
            if (i) oss << ", ";
            auto feature = features[i];
            oss << cpu_feature_str(feature) << "=" << (check_cpu_feature(feature) ? "ok" : "failed");
        }
        oss << "}";
        return oss.str();
    }

    static bool check_cpu_features() {
        //TestCPUFeature
        std::vector<CPUFeature> features;
#if defined(TS_USE_FMA)
        features.emplace_back(FMA);
#endif
#if defined(TS_USE_AVX)
        features.emplace_back(AVX);
#elif defined(TS_USE_SSE)
        features.emplace_back(SSE);
        features.emplace_back(SSE2);
#endif
        for (auto &fea : features) {
            auto flag = check_cpu_feature(fea);
            if (!flag) {
                /*TS_LOG_ERROR << "The processor does not support the current instruction set: " << feature_log(features)
                             << eject;*/
                return false;
            }
        }
        return true;
    }

    Workbench::Workbench(const ComputingDevice &device) {
        //check_cpu_features();

        this->m_device_context.initialize(device);
        auto &memory_device = this->m_device_context.memory_device;

        this->m_static_memory = DynamicSyncMemoryController::Make(memory_device, true);
        // TODO: Make real flow memory controller
        this->m_flow_memory = HypeSyncMemoryController<FlowMemoryController>::Make(memory_device, false);
        this->m_dynamic_memory = DynamicSyncMemoryController::Make(memory_device, false);
        this->m_stack = std::make_shared<Stack>(memory_device, this->m_flow_memory);
        // bind flow and dynamic memory, so you can use it to alloc memory in any where
        this->m_runtime_context.bind_flow(this->m_flow_memory);
        this->m_runtime_context.bind_dynamic(this->m_dynamic_memory);

        this->m_switch_controller = std::make_shared<SwitchControll>();
        if(!check_cpu_features()){
            m_switch_controller->auto_switch(device);
        }
    }

    Workbench::Workbench(const ComputingDevice &device, int computing_thread_number)
            : self(device) {
        this->m_runtime_context.set_computing_thread_number(computing_thread_number);
    }

    Workbench::Workbench(const ComputingDevice &device, CpuEnable::CpuPowerMode cpu_mode)
            : self(device) {
        set_cpu_power_mode(cpu_mode);
    }

    Workbench::~Workbench() {
        this->m_desktop.reset();
        this->m_stack->clear();
        this->m_inputs.clear();
        this->m_outputs.clear();
        this->m_device_context.finalize();
        std::stack<ProgramEnv> empty;
        this->m_env.swap(empty);
        this->m_switch_controller.reset();
    }

    static inline std::unique_ptr<ctx::bind<Profiler>> bind_profiler(bool _do, Profiler &profiler) {
        if (!_do) return nullptr;
        return std::unique_ptr<ctx::bind<Profiler>>(new ctx::bind<Profiler>(profiler));
    }

    void Workbench::run() {
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }

        this->m_hooked_tensor.clear();

        auto outputs = launch_offline(m_desktop, m_inputs);

        m_outputs = outputs;
    }

    Workbench::shared Workbench::clone() const {
        Workbench::shared dolly(new Workbench(
                this->m_device_context.computing_device));

        BindWorkbenchRuntime _bind_runtime(*dolly);

        dolly->m_inputs.resize(this->m_inputs.size());
        dolly->m_outputs.resize(this->m_outputs.size());
        dolly->m_runtime_context = this->m_runtime_context.clone();
        if (this->m_desktop) {
            dolly->m_desktop = this->m_desktop->clone();
        }

        return std::move(dolly);
    }

    Workbench::shared Workbench::Load(const Module::shared &module, const ComputingDevice &device) {
        auto bench = std::make_shared<Workbench>(device);
        bench->setup(bench->compile(module));
        return bench;
    }

    void Workbench::jump_relative(int shift) {
        this->m_env.top().pointer += shift;
    }

    void Workbench::jump_absolute(size_t pointer) {
        this->m_env.top().pointer = pointer;
    }

    void Workbench::input(int slot, const Tensor &tensor) {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        m_inputs[slot] = tensor;
    }

    const Tensor &Workbench::output(int slot) const {
        if (slot < 0 || size_t(slot) >= m_outputs.size()) {
            TS_LOG_ERROR << "Output index out of range. with index=" << slot << eject;
        }
        return m_outputs[slot];
    }

    int Workbench::input_count() const {
        return int(m_inputs.size());
    }

    int Workbench::output_count() const {
        return int(m_outputs.size());
    }

    const Tensor &Workbench::input(int slot) const {
        if (slot < 0 || size_t(slot) >= m_inputs.size()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        return m_inputs[slot];
    }

    void Workbench::bind_filter(const std::string &name, ImageFilter::shared filter) {
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }
        this->bind_filter(m_desktop->input_slot(name), std::move(filter));
    }

    void Workbench::bind_filter(int slot, ImageFilter::shared filter) {
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }
        if (slot < 0 || slot >= m_desktop->input_count()) {
            TS_LOG_ERROR << "Input index out of range. with index=" << slot << eject;
        }
        BindWorkbenchRuntime _bind_runtime(*this);
        filter->compile();
        m_desktop->bind_filter(slot, filter->program());
    }

    void Workbench::input(const std::string &name, const Tensor &tensor) {
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }
        this->input(m_desktop->input_slot(name), tensor);
    }

    const Tensor &Workbench::output(const std::string &name) const {
        {
            auto it = m_hooked_tensor.find(name);
            if (it != m_hooked_tensor.end()) {
                return it->second;
            }
        }
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }
        return this->output(m_desktop->output_slot(name));
    }

    const Tensor &Workbench::input(const std::string &name) const {
        if (m_desktop == nullptr) {
            TS_LOG_ERROR << "Can not run workbench with no program setup" << eject;
        }
        return this->input(m_desktop->input_slot(name));
    }

    void Workbench::push_data_segment(int data_index) {
        // TODO: deal with data_sagment, in case of thread sharing, waitting testing
        this->m_stack->push(this->m_env.top().program->data_segment(data_index));
    }

    Operator::shared Workbench::offline_create(const Bubble &bubble, bool strict) {
        BindWorkbenchRuntime _bind_runtime(*this);

        auto built_op = OperatorCreator::Create(m_device_context.computing_device.type(), bubble.op(), strict);

        if (built_op == nullptr) return nullptr;

        for (auto &param : bubble.params()) {
            built_op->set(param.first, param.second);
        }

        built_op->init();

        return built_op;
    }

    void Workbench::offline_run(Operator::shared op, const std::vector<Tensor> &input, std::vector<Tensor> &output) {
        // Stack stack(m_device_context.memory_device, m_dynamic_memory);
        m_stack->push_base(int(m_stack->size())); // empty base
        need pop_base(&Stack::pop_base, m_stack.get());
        need clear_stack(&Stack::clear, m_stack.get());
        auto &stack = *m_stack;

        BindWorkbenchRuntime _bind_runtime(*this);

        for (auto &tensor : input) {
            stack.push(tensor);
        }

        auto output_count = RunOperator(op, stack, int(input.size()));

        TS_AUTO_CHECK(output_count == stack.size());

        output.resize(output_count);

        for (int i = 0; i < output_count; ++i) {
            output[i] = stack[i];
        }
    }

    void Workbench::offline_infer(Operator::shared op, const std::vector<Tensor> &input,
                                  std::vector<Tensor::Prototype> &output) {
        // Stack stack(m_device_context.memory_device, m_dynamic_memory);
        m_stack->push_base(int(m_stack->size())); // empty base
        need pop_base(&Stack::pop_base, m_stack.get());
        need clear_stack(&Stack::clear, m_stack.get());
        auto &stack = *m_stack;

        BindWorkbenchRuntime _bind_runtime(*this);

        for (auto &tensor : input) {
            stack.push(tensor);
        }

        op->infer(stack, output);
    }

    void Workbench::set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value) {
        if (m_desktop == nullptr) return;

        BindWorkbenchRuntime _bind_runtime(*this);

        m_desktop->set_operator_param(node_name, param, value);
    }

    void Workbench::cast_tensor(DTYPE dtype) {
        if (m_cast_op == nullptr) {
            m_cast_op = OperatorCreator::Create(
                    m_device_context.computing_device.type(),
                    name::layer::cast(), false);
        }
        auto *cast_op = dynamic_cast<base::CastV2*>(m_cast_op.get());
        if (cast_op != nullptr) {
            cast_op->set_dtype(dtype);
        } else {
            m_cast_op->set(name::dtype, tensor::from<int32_t>(dtype));
            m_cast_op->init();
        }
        TS_AUTO_CHECK(1 == RunOperator(m_cast_op, *m_stack, 1));
    }

    Operator::shared Workbench::online_create(const Bubble &bubble, bool strict) {
        return offline_create(bubble, strict);
    }

    int Workbench::online_run(Operator::shared op, const std::vector<Tensor> &input) {
        // m_stack->push_base(m_stack->size()); // empty base
        // need pop_base(&Stack::pop_base, m_stack.get());
        for (auto &tensor : input) {
            m_stack->push(tensor);
        }
        return online_run(op, int(stack().size()));
    }

    int Workbench::online_run(Operator::shared op, int argc) {
        BindWorkbenchRuntime _bind_runtime(*this);

        return RunOperator(op, *m_stack, argc);
    }

    void Workbench::online_run(Instruction::shared inst) {
        BindWorkbenchRuntime _bind_runtime(*this);

        inst->run(*this);
    }

    void Workbench::online_run(Instruction::shared inst, const std::vector<Tensor> &input) {
        // m_stack->push_base(m_stack->size()); // empty base
        // need pop_base(&Stack::pop_base, m_stack.get());
        for (auto &tensor : input) {
            m_stack->push(tensor);
        }
        online_run(inst);
    }

    int Workbench::online_run(const Bubble &bubble, int argc, bool strict) {
        return online_run(online_create(bubble, strict), argc);
    }

    int Workbench::online_run(const Bubble &bubble, const std::vector<Tensor> &input, bool strict) {
        return online_run(online_create(bubble, strict), input);
    }

    static Tensor adjust_nhwc(const Tensor &tensor) {
        if (tensor.dims() < 2 || tensor.dims() > 4) {
            TS_LOG_ERROR << "Can not filter input with shape: " << to_string(tensor.sizes()) << eject;
        }
        if (tensor.dims() == 2) {
            return tensor.reshape({1, tensor.size(0), tensor.size(1), 1});
        }
        if (tensor.dims() == 3) {
            return tensor.reshape({1, tensor.size(0), tensor.size(1), tensor.size(2)});
        }
        return tensor;
    }

    int Workbench::launch_online(Program::shared program, int nargs) {
        if (program == nullptr) {
            TS_LOG_ERROR << "Can not launch null program." << eject;
        }
        if (program->input_count() != nargs) {
            TS_LOG_ERROR << "nargs must be " << program->input_count() << " vs. " << nargs << " got." << eject;
        }
        if (m_stack->size() < nargs) {
            TS_LOG_ERROR << "stack must have arguments at less " << nargs << " vs. " << m_stack->size() << " got." << eject;
        }
        if (program->device() != this->device().computing_device) {
            TS_LOG_ERROR << "Running " << program->device() << " on " << this->device().computing_device;
        }

        m_env.push(ProgramEnv(program));
        ts::need pop_env(&std::stack<ProgramEnv>::pop, &m_env);

        /**
         * bind profiler for running
         */
        auto _bind_profiler = bind_profiler(m_do_profile && ctx::get<Profiler>() != &m_profiler, m_profiler);

        /**
         * bind context
         * TODO: had to find way to avoid loop binding
         */
        BindWorkbenchRuntime _bind_runtime(*this);

        /**
         * Save base, so now can do something
         */
        this->m_stack->push_base(-nargs);
        ts::need pop_base(&Stack::pop_base, this->m_stack.get());

        /**
         * do filter and cast dtype
         */
        for (int i = 0; i < nargs; ++i) {
            auto &arg = *this->m_stack->index(-nargs);

            if (arg.device() == this->device().memory_device) {
                this->m_stack->push(arg);
            } else {
                this->m_stack->clone_push(arg); // in case of sync memory out of flow memory
            }

            // top is the arg i

            auto filter = this->m_env.top().program->input_filter(i);
            if (filter != nullptr) {
                // ajust to nhwc image format
                this->m_stack->push(adjust_nhwc(arg));
                this->m_stack->erase(-2);   // delete arg before
                launch_online(filter, 1);
            }
            auto dtype = this->m_env.top().program->input_dtype(i);
            if (dtype == VOID) continue;
            cast_tensor(dtype);
        }
        /**
         * build input for program
         */
        this->m_stack->erase(0, nargs);

        /**
         * Start run program
         */
        while (true) {
            auto &running_program = this->m_env.top();
            auto &pointer = running_program.pointer;
            auto &length = running_program.length;
            if (pointer >= length) break;
            auto &inst = running_program.program->instruction(pointer);
            pointer++;
            inst->run(*this);
        }

        /**
         * Check output
         */
        if (this->m_stack->size() != program->output_count()) {
            TS_LOG_ERROR << "Got unexpected output number want " << program->output_count() << " vs. "
                         << this->m_stack->size() << " given." << eject;
        }

        return int(this->m_stack->size());
    }

    void Workbench::setup(Program::shared program) {
        this->m_desktop = program;
        if (program == nullptr) {
            this->m_inputs.clear();
            this->m_outputs.clear();
        } else {
            this->m_inputs.resize(program->input_count());
            this->m_outputs.resize(program->output_count());
        }
        this->m_hooked_tensor.clear();
    }

    std::vector<Tensor> Workbench::launch_offline(Program::shared program, const std::vector<Tensor> &args) {
        /**
         * save stack
         */
        this->m_stack->push_base(int(this->m_stack->size()));
        ts::need pop_base(&Stack::pop_base, this->m_stack.get());
        ts::need clear_stack(&Stack::clear, this->m_stack.get());

        for (auto &arg : args) {
            this->m_stack->push(arg);
        }

        int return_count = launch_online(program, int(args.size()));

        std::vector<Tensor> outputs;
        for (int i = 0; i < return_count; ++i) {
            outputs.emplace_back(*this->m_stack->index(i));
        }

        return std::move(outputs);
    }

    std::vector<Tensor> Workbench::launch_offline(Program::shared program, const std::map<std::string, Tensor> &args) {
        auto nargs = int(args.size());
        if (program->input_count() != nargs) {
            TS_LOG_ERROR << "nargs must be " << program->input_count() << " vs. " << nargs << " got." << eject;
        }

        std::vector<Tensor> list_args(nargs);

        for (auto &pair : args) {
            auto &name = pair.first;
            auto &value = pair.second;
            auto slot = program->input_slot(name);
            list_args[slot] = value;
        }

        return launch_offline(program, list_args);
    }

    Program::shared Workbench::compile(const Module::shared &module) {
        /**
         * tell compiler, who is compiling
         */
        BindWorkbenchRuntime _bind_runtime(*this);

        /**
         * do compile, from module to program
         */
        return Program::Compile(module, this->device().computing_device);
    }

    void Workbench::setup_runtime() {
        // this->runtime().setup_context();
        this->setup_context();
    }

    void Workbench::setup_device() {
        // DeviceContext::Switch(&this->device());
        this->setup_context();
    }

    Program::shared Workbench::compile(const Module::shared &module, const std::string &options) {
        BindWorkbenchRuntime _bind_runtime(*this);
        return Program::Compile(module, this->device().computing_device, options);
    }

    Workbench::shared
    Workbench::Load(const Module::shared &module, const ComputingDevice &device, const std::string &options) {
        auto bench = std::make_shared<Workbench>(device);
        bench->setup(bench->compile(module, options));
        return bench;
    }

    void Workbench::run_hook(const std::vector<std::string> &node_names) {
        this->m_hooked_tensor.clear();

        auto controller = std::make_shared<DynamicMemoryController>(MemoryDevice(CPU));

        // Save input
        auto &input_names = m_desktop->input_names();
        auto input_count = m_desktop->input_count();
        for (int i = 0; i < input_count; ++i) {
            auto &value = m_inputs[i];
            auto &name = input_names[i];
            auto name_it = m_hooked_tensor.find(name);
            if (name_it == m_hooked_tensor.end()) {
                m_hooked_tensor.insert(std::make_pair(name, value.clone(controller)));
            } else {
                name_it->second =value.clone(controller);
            }
        }

        std::unordered_set<std::string> node_name_set;
        for (auto &node_name : node_names) {
            node_name_set.insert(node_name);
        }

        Hook hooker;
        hooker.after_run([&](const Hook::StructAfterRun &info) {
            auto &op = *info.op;
            auto &stack = *info.stack;
            auto name = op.name();
            auto it = node_name_set.find(name);
            if (it == node_name_set.end()) return;
            if (stack.size() < 1) return;
            auto &value = stack[0];
            auto name_it = m_hooked_tensor.find(name);
            if (name_it == m_hooked_tensor.end()) {
                m_hooked_tensor.insert(std::make_pair(name, value.clone(controller)));
            } else {
                name_it->second =value.clone(controller);
            }
        });
        ctx::bind<Hook> _hook(hooker);

        this->run();
    }

    const std::string &Workbench::summary() {
        uint64_t shared_memory = 0;
        if (m_desktop) {
            auto &stack = m_desktop->data_segment();
            auto size = stack.size();
            for (size_t i = 0; i < size; ++i) {
                auto &tensor = stack[i];
                shared_memory += tensor.count() * tensor.proto().type_bytes();
            }
        }

        std::ostringstream oss;
        oss << "{\"device\": \"" << m_device_context.computing_device << "\""
            << ", \"thread\": " << m_runtime_context.get_computing_thread_number()
            << ", \"shared\": \"" << memory_size_string(shared_memory) << "\""
            << ", \"memory\": " << m_flow_memory->summary()
            << "}";
        m_summary = oss.str();
        return m_summary;
    }

    bool Workbench::set_cpu_power_mode(CpuEnable::CpuPowerMode cpu_mode){
        bool flag = CpuEnable::set_power_mode(cpu_mode);
        if (flag) {
            int fixed_thread_num = this->m_runtime_context.get_computing_thread_number();
            switch (cpu_mode)
            {
            case ts::CpuEnable::BALANCE:
                fixed_thread_num = CpuEnable::get_cpu_num();
                break;
            case ts::CpuEnable::BIGCORE:
                fixed_thread_num = CpuEnable::get_cpu_big_num();
                break;
            case ts::CpuEnable::LITTLECORE:
                fixed_thread_num = CpuEnable::get_cpu_little_num();
                break;
            default:
                break;
            }
            this->m_runtime_context.set_computing_thread_number(fixed_thread_num);
        }
        return flag;
    }

    SwitchControll::shared Workbench::switch_controller(){
        return m_switch_controller;
    }

}

TS_LITE_CONTEXT(ts::Workbench)
