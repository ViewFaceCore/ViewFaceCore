//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_OPERATOR_H
#define TENSORSTACK_THIRD_DRAGON_OPERATOR_H

#include "workspace.h"
#include "tensor.h"

#include <set>

#include <core/tensor_builder.h>

namespace ts {
    namespace dragon {
        using OperatorDef = ts::Operator *;

        class OperatorBase {
        public:
            explicit OperatorBase(const OperatorDef &def, Workspace *ws) : m_def(def), m_ws(ws) {
                m_type = m_def->op();
            }

            template<typename T>
            T Arg(const std::string &name, const T &default_value) {
                if (!m_def->has(name)) return default_value;
                auto value = tensor::cast(dtypeid<T>::id, m_def->get(name));
                if (value.count() < 1) return default_value;
                return value.template data<T>(0);
            }


            template<typename T>
            std::vector<T> Args(const std::string &name) {
                if (!m_def->has(name)) return std::vector<T>();
                auto value = tensor::cast(dtypeid<T>::id, m_def->get(name));
                auto data = value.template data<T>();
                auto count = value.count();
                return std::vector<T>(data, data + count);
            }

            virtual void RunOnDevice() = 0;

            void bind_inputs(const std::vector<ts::Tensor> &inputs) {
                m_inputs = std::vector<dragon::Tensor>(inputs.begin(), inputs.end());
            }

            void bind_outputs(const std::vector<ts::Tensor> &outputs) {
                m_outputs = std::vector<dragon::Tensor>(outputs.begin(), outputs.end());
            }

            void clear_inputs() { m_inputs.clear(); }

            void clear_outputs() { m_outputs.clear(); }

            Tensor Input(int i) {
                if (i >= 0) return m_inputs[i];
                return m_inputs[int(m_inputs.size()) + i];
            }

            Tensor *Output(int i) {
                if (i >= 0) return &m_outputs[i];
                return &m_outputs[int(m_outputs.size()) + i];
            }

            size_t InputSize() const {
                return m_inputs.size();
            }

            size_t OutputSize() const {
                return m_outputs.size();
            }

            void bind_outputs(int N) {
                m_outputs.resize(N);
            }

            std::vector<ts::Tensor> outputs() const {
                return std::vector<ts::Tensor>(m_outputs.begin(), m_outputs.end());
            }

            Workspace *ws() { return m_ws; }

            std::string DTypeHelper(
                    const Tensor &tensor,
                    const std::set<std::string> &dtypes) const {
                std::stringstream ss;
                ss << "Unsupported DType of Input(" << tensor.name() << "): "
                   << TypeMetaToString(tensor.meta()) << "\n";
                ss << "<" << type() << "Op>" << " supports the following dtypes: {\n";
                for (auto &dtype : dtypes) ss << "    * " << dtype << ",\n";
                ss << "}";
                return ss.str();
            }

            std::string DTypeHelper(
                    const std::string &dtype,
                    const std::set<std::string> &dtypes) const {
                std::stringstream ss;
                ss << "Unsupported DType: " << dtype << "\n";
                ss << "<" << type() << "Op>" << " supports the following dtypes: {\n";
                for (auto &dtype : dtypes) ss << "    * " << dtype << ",\n";
                ss << "}";
                return ss.str();
            }

            const std::string &type() const { return m_type; }

            void temp(dragon::Tensor *tmp) { m_temp.insert(tmp); }

            void temp(dragon::Tensor &tmp) { return temp(&tmp); }

            void erase_temp(dragon::Tensor *tmp) { m_temp.erase(tmp); }

            void erase_temp(dragon::Tensor &tmp) { erase_temp(&tmp); }

            void clear_temp() { m_temp.clear(); }

            void clean() {
                m_inputs.clear();
                m_outputs.clear();
                for (auto &tmp : m_temp) {
                    tmp->dispose();
                }
            }

        private:
            OperatorDef m_def;
            Workspace *m_ws;
            std::vector<Tensor> m_inputs;
            std::vector<Tensor> m_outputs;
            std::string m_type;

            std::set<dragon::Tensor*> m_temp;
        };

        template<typename Context>
        class Operator : public OperatorBase {
        public:
            using self = Operator;
            using supper = OperatorBase;

            Operator(const OperatorDef &def, Workspace *ws) : supper(def, ws) {
            }

            Context *ctx() { return nullptr; }
        };

#define USE_OPERATOR_FUNCTIONS \
        using OperatorBase::Input; \
        using OperatorBase::Output; \
        using OperatorBase::InputSize; \
        using OperatorBase::OutputSize; \
        using OperatorBase::DTypeHelper; \
        using OperatorBase::ws; \
        using OperatorBase::clear_inputs; \
        using OperatorBase::clear_outputs; \
        using OperatorBase::temp; \
        using Operator<Context>::ctx
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_OPERATOR_H
