//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_CPP_WORKBENCH_H
#define TENNIS_API_CPP_WORKBENCH_H

#include "../workbench.h"

#include "except.h"
#include "device.h"
#include "module.h"
#include "tensor.h"
#include "image_filter.h"
#include "program.h"

#include <string>

namespace ts {
    namespace api {
        enum class CpuPowerMode : int32_t {
            BALANCE = TS_CPU_BALANCE,
            BIG_CORE = TS_CPU_BIG_CORE,
            LITTLE_CORE = TS_CPU_LITTLE_CORE,
        };

        /**
         * @see ts_Workbench
         */
        class Workbench {
        public:
            using self = Workbench;
            using raw = ts_Workbench;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            static self NewRef(raw *ptr) { return self(ptr); }

            Workbench(const self &) = default;

            Workbench &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            bool operator==(std::nullptr_t) const { return get_raw() == nullptr; }

            bool operator!=(std::nullptr_t) const { return get_raw() != nullptr; }

            // Workbench() : self((ts_Device*)(nullptr)) {}

            Workbench(std::nullptr_t) {}

            Workbench() = default;

            explicit Workbench(const Device &device) : self(device.get_raw()) {}

            explicit Workbench(const ts_Device *device)
                : self(ts_new_Workbench(device)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            static Workbench Load(const Module &module, const Device &device) {
                return Load(module.get_raw(), device.get_raw());
            }

            static Workbench Load(const ts_Module *module, const Device &device) {
                return Load(module, device.get_raw());
            }

            static Workbench Load(const Module &module, const ts_Device *device) {
                return Load(module.get_raw(), device);
            }

            static Workbench Load(const ts_Module *module, const ts_Device *device) {
                Workbench loaded(ts_Workbench_Load(module, device));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            Workbench clone() const {
                Workbench dolly(ts_Workbench_clone(m_impl.get()));
                TS_API_AUTO_CHECK(dolly.m_impl != nullptr);
                return std::move(dolly);
            }

            void input(int slot, const ts_Tensor *tensor) {
                TS_API_AUTO_CHECK(ts_Workbench_input(m_impl.get(), slot, tensor));
            }

            void input(int slot, const Tensor &tensor) {
                input(slot, tensor.get_raw());
            }

            void input(const std::string &name, const ts_Tensor *tensor) {
                TS_API_AUTO_CHECK(ts_Workbench_input_by_name(m_impl.get(), name.c_str(), tensor));
            }

            void input(const std::string &name, const Tensor &tensor) {
                input(name, tensor.get_raw());
            }

            void run() {
                TS_API_AUTO_CHECK(ts_Workbench_run(m_impl.get()));
            }

            void output(int slot, ts_Tensor *tensor) {
                TS_API_AUTO_CHECK(ts_Workbench_output(m_impl.get(), slot, tensor));
            }

            void output(int slot, const Tensor &tensor) {
                output(slot, tensor.get_raw());
            }

            void output(const std::string &name, ts_Tensor *tensor) {
                TS_API_AUTO_CHECK(ts_Workbench_output_by_name(m_impl.get(), name.c_str(), tensor));
            }

            void output(const std::string &name, Tensor &tensor) {
                output(name, tensor.get_raw());
            }

            Tensor output(int slot) {
                Tensor tensor;
                output(slot, tensor);
                return std::move(tensor);
            }

            Tensor output(const std::string &name) {
                Tensor tensor;
                output(name, tensor);
                return std::move(tensor);
            }

            void set_computing_thread_number(int number) {
                TS_API_AUTO_CHECK(ts_Workbench_set_computing_thread_number(m_impl.get(), number));
            }

            void bind_filter(int slot, const ts_ImageFilter *filter) {
                TS_API_AUTO_CHECK(ts_Workbench_bind_filter(m_impl.get(), slot, filter));
            }

            void bind_filter(int slot, const ImageFilter &tensor) {
                bind_filter(slot, tensor.get_raw());
            }

            void bind_filter(const std::string &name, const ts_ImageFilter *filter) {
                TS_API_AUTO_CHECK(ts_Workbench_bind_filter_by_name(m_impl.get(), name.c_str(), filter));
            }

            void bind_filter(const std::string &name, const ImageFilter &tensor) {
                bind_filter(name, tensor.get_raw());
            }

            void setup_context() const {
                TS_API_AUTO_CHECK(ts_Workbench_setup_context(m_impl.get()));
            }

            void setup(const ts_Program *program) {
                TS_API_AUTO_CHECK(ts_Workbench_setup(m_impl.get(), program));
            }

            void setup(const Program &program) {
                TS_API_AUTO_CHECK(ts_Workbench_setup(m_impl.get(), program.get_raw()));
            }

            Program compile(const Module &module) {
                return compile(module.get_raw());
            }

            Program compile(const ts_Module *module) {
                auto program = Program::NewRef(ts_Workbench_compile(m_impl.get(), module));
                TS_API_AUTO_CHECK(program != nullptr);
                return std::move(program);
            }

            void setup_device() const {
                TS_API_AUTO_CHECK(ts_Workbench_setup_device(m_impl.get()));
            }

            void setup_runtime() const {
                TS_API_AUTO_CHECK(ts_Workbench_setup_runtime(m_impl.get()));
            }

            int input_count() const {
                return ts_Workbench_input_count(m_impl.get());
            }

            int output_count() const {
                return ts_Workbench_output_count(m_impl.get());
            }

            void run_hook(const std::vector<std::string> &node_names) {
                std::vector<const char *> c_node_names(node_names.size());
                for (size_t i = 0; i < node_names.size(); ++i) {
                    c_node_names[i] = node_names[i].c_str();
                }
                TS_API_AUTO_CHECK(
                        ts_Workbench_run_hook(m_impl.get(), c_node_names.data(), int32_t(c_node_names.size())));
            }

            static Workbench Load(const Module &module, const Device &device, const std::string &options) {
                Workbench loaded(ts_Workbench_Load_v2(module.get_raw(), device.get_raw(), options.c_str()));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            Program compile(const Module &module, const std::string &options) {
                auto program = Program::NewRef(ts_Workbench_compile_v2(m_impl.get(), module.get_raw(), options.c_str()));
                TS_API_AUTO_CHECK(program != nullptr);
                return std::move(program);
            }

            void set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value) {
                TS_API_AUTO_CHECK(ts_Workbench_set_operator_param(
                        m_impl.get(), node_name.c_str(), param.c_str(), value.get_raw()))
            }

            std::string summary() const {
                auto json = ts_Workbench_summary(m_impl.get());
                TS_API_AUTO_CHECK(json != nullptr);
                return json;
            }

            bool set_cpu_mode(ts_CpuPowerMode mode) {
                return ts_Workbench_set_cpu_mode(m_impl.get(), mode);
            }

            bool set_cpu_mode(CpuPowerMode mode) {
                return ts_Workbench_set_cpu_mode(m_impl.get(), ts_CpuPowerMode(mode));
            }

        private:
            Workbench(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Workbench); }

            shared_raw m_impl;
        };
    }
}

#endif //TENNIS_API_CPP_WORKBENCH_H
