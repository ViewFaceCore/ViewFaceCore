//
// Created by kier on 19-5-7.
//

#ifndef TENNIS_API_CPP_PROGORAM_H
#define TENNIS_API_CPP_PROGORAM_H

#include "../program.h"

#include "except.h"
#include "device.h"
#include "module.h"

#include <string>

namespace ts {
    namespace api {
        /**
         * @see ts_Program
         */
        class Program {
        public:
            using self = Program;
            using raw = ts_Program;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            static self NewRef(raw *ptr) { return self(ptr); }

            Program(const self &) = default;

            Program &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            bool operator==(std::nullptr_t) const { return get_raw() == nullptr; }

            bool operator!=(std::nullptr_t) const { return get_raw() != nullptr; }

            Program(std::nullptr_t) {}

            Program() = default;

            static Program Compile(const Module &module, const Device &device) {
                return Compile(module.get_raw(), device.get_raw());
            }

            static Program Compile(const ts_Module *module, const Device &device) {
                return Compile(module, device.get_raw());
            }

            static Program Compile(const Module &module, const ts_Device *device) {
                return Compile(module.get_raw(), device);
            }

            static Program Compile(const ts_Module *module, const ts_Device *device) {
                Program loaded(ts_Program_Compile(module, device));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            static Program Compile(const Module &module, const Device &device, const std::string &options) {
                Program loaded(ts_Program_Compile_v2(module.get_raw(), device.get_raw(), options.c_str()));
                TS_API_AUTO_CHECK(loaded.m_impl != nullptr);
                return std::move(loaded);
            }

            Program clone() const {
                Program dolly(ts_Program_clone(m_impl.get()));
                TS_API_AUTO_CHECK(dolly.m_impl != nullptr);
                return std::move(dolly);
            }

            int input_count() const {
                return ts_Program_input_count(m_impl.get());
            }

            int output_count() const {
                return ts_Program_output_count(m_impl.get());
            }

            void set_operator_param(const std::string &node_name, const std::string &param, const Tensor &value) {
                TS_API_AUTO_CHECK(ts_Program_set_operator_param(
                        m_impl.get(), node_name.c_str(), param.c_str(), value.get_raw()))
            }

        private:
            Program(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_Program); }

            shared_raw m_impl;
        };
    }
}

#endif //TENNIS_API_CPP_PROGORAM_H
