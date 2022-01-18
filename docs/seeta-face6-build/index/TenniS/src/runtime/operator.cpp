//
// Created by kier on 2018/5/19.
//

#include "runtime/operator.h"
#include "utils/box.h"

#include "utils/assert.h"
#include "core/tensor_builder.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"

#include <climits>
#include <runtime/operator.h>

#include "utils/need.h"
#include "runtime/stack.h"
#include "module/bubble.h"


namespace ts {
    bool Operator::has(const std::string &param) const {
        return this->m_params.find(param) != this->m_params.end();
    }

    void Operator::set(const std::string &param, const Tensor &value) {
        bool is_retention_param = !param.empty() && param[0] == retention_param_sign;
        if (!is_retention_param && !is_in_fields(param) && m_param_checking_mode == ParamCheckingMode::STRICT) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            this->m_params.insert(std::make_pair(param, value));
        } else {
            param_it->second = value;
        }
    }

    Tensor &Operator::get(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        return param_it->second;
    }

    const Tensor &Operator::get(const std::string &param) const {
        return const_cast<self *>(this)->get(param);
    }

    void Operator::clear(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        this->m_params.erase(param_it);
    }

    void Operator::clear_params() {
        std::vector<std::pair<std::string, Tensor>> retention_params;
        for (auto &param_tenosr_pair : m_params) {
            auto &param = param_tenosr_pair.first;
            bool is_retention_param = !param.empty() && param[0] == retention_param_sign;
            if (is_retention_param) {
                retention_params.emplace_back(param_tenosr_pair);
            }
        }
        m_params.clear();
        m_params.insert(retention_params.begin(), retention_params.end());
    }

    void Operator::clear_fields() {
        this->m_optional_fields.clear();
        for (auto &param : this->m_required_fields) {
            this->m_params.erase(param);
        }
        this->m_required_fields.clear();
    }

    void Operator::field(const std::string &param, Operator::FieldAttr attr, const Tensor &default_value) {
        this->field(param, attr);
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            this->m_params.insert(std::make_pair(param, default_value));
        } else {
            param_it->second = default_value;
        }
    }

    void Operator::field(const std::string &param, Operator::FieldAttr attr) {
        switch (attr) {
            default:
                break;
            case OPTIONAL:
                this->m_optional_fields.insert(param);
                break;
            case REQUIRED:
                this->m_required_fields.insert(param);
                this->m_params.erase(param);
                break;
        }
    }

    std::vector<std::string> Operator::unsatisfied_fields() const {
        std::vector<std::string> fileds;
        for (auto &param : this->m_required_fields) {
            auto param_it = this->m_params.find(param);
            if (param_it == this->m_params.end() || param_it->second.empty()) {
                fileds.push_back(param);
            }
        }
        return fileds;
    }

    bool Operator::check_params() const {
        if (m_param_checking_mode != ParamCheckingMode::STRICT) return true;
        for (auto &param : this->m_required_fields) {
            auto param_it = this->m_params.find(param);
            if (param_it == this->m_params.end() || param_it->second.empty()) {
                return false;
            }
        }
        return true;
    }

    std::string Operator::fuzzy_field_name(const std::string &name) {
        if (m_required_fields.empty() || m_optional_fields.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &target_name : m_required_fields) {
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        for (auto &target_name : m_optional_fields) {
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    std::string Operator::fuzzy_param_name(const std::string &name) {
        if (m_params.empty()) return "";
        int min_edit_distance = INT_MAX;
        std::string closest_name;
        for (auto &param_tensor_pair : m_params) {
            auto &target_name = param_tensor_pair.first;
            int dist = edit_distance(name, target_name);
            if (dist < min_edit_distance) {
                closest_name = target_name;
                min_edit_distance = dist;
            }
        }
        return closest_name;
    }

    bool Operator::is_in_fields(const std::string &name) {
        return m_optional_fields.find(name) != m_optional_fields.end() ||
               m_required_fields.find(name) != m_required_fields.end();
    }

    bool Operator::is_in_params(const std::string &name) {
        return m_params.find(name) != m_params.end();
    }

    void Operator::init() {
        if (!this->check_params()) {
            std::ostringstream oss;
            auto unsatisfied_fields = this->unsatisfied_fields();
            std::string op, name;
            try {
                op = tensor::to_string(get("#op"));
                name = tensor::to_string(get("#name"));
            } catch (const Exception &) {

            }
            oss << "Operator " << op << " \"" << name << "\" has unsatisfied fields: ";
            for (size_t i = 0; i < unsatisfied_fields.size(); ++i) {
                if (i) oss << ", ";
                oss << "\"" << unsatisfied_fields[i] << "\"";
            }

            TS_LOG_ERROR(oss.str()) << eject;
        }
    }

    MemoryDevice Operator::memory_device() const {
        auto device = ctx::ptr<DeviceContext>();
        return device ? device->memory_device : MemoryDevice();
    }

    ComputingDevice Operator::computing_device() const {
        auto device = ctx::ptr<DeviceContext>();
        return device ? device->computing_device : ComputingDevice();
    }

    const Operator::hash_map<std::string, Tensor> &Operator::params() const {
        return m_params;
    }

    std::string Operator::op() const {
        auto &param = Bubble::RetentionParam::op;
        return has(param) ? tensor::to_string(get(param)) : std::string();
    }

    std::string Operator::name() const {
        auto &param = Bubble::RetentionParam::name;
        return has(param) ? tensor::to_string(get(param)) : std::string();
    }

    int Operator::output_count() const {
        return 1;
    }

    TensorPrototype Operator::infer(Stack &stack) {
        TensorPrototype proto;
        std::vector<Tensor::Prototype> fields;
        infer(stack, fields);
        proto.pack(fields);
        return std::move(proto);
    }

    void Operator::set_param_checking_mode(Operator::ParamCheckingMode mode) {
        this->m_param_checking_mode = mode;
    }

    std::vector<std::string> Operator::list_all_fields() const {
        std::vector<std::string> result;
        for (auto &f : m_required_fields) {
            result.emplace_back(f);
        }
        for (auto &f : m_optional_fields) {
            result.emplace_back(f);
        }
        return result;
    }

    std::vector<std::string> Operator::list_required_fields() const {
        std::vector<std::string> result;
        for (auto &f : m_required_fields) {
            result.emplace_back(f);
        }
        return result;
    }

    std::vector<std::string> Operator::list_optional_fields() const {
        std::vector<std::string> result;
        for (auto &f : m_optional_fields) {
            result.emplace_back(f);
        }
        return result;
    }

    int RunOperator(Operator::shared op, Stack &stack, int nargs) {
        TS_AUTO_CHECK(stack.size() >= static_cast<size_t>(nargs));

        // save base
        stack.push_base(-nargs);
        ts::need pop_base(&Stack::pop_base, &stack);

        // call function
        auto return_size = op->run(stack);

        TS_AUTO_CHECK(stack.size() >= static_cast<size_t>(return_size));    // must have enough returns

        // add base
        stack.erase(0, -return_size);

        return return_size;
    }

    int InferOperator(Operator::shared op, Stack &stack, int nargs, std::vector<Tensor::Prototype> &output) {
        TS_AUTO_CHECK(stack.size() >= static_cast<size_t>(nargs));

        // save base
        stack.push_base(-nargs);
        ts::need pop_base(&Stack::pop_base, &stack);

        auto returned_value = op->infer(stack, output);

        stack.clear();  // pop all arguments

        return returned_value;
    }
}
