//
// Created by kier on 2019/1/23.
//

#include "module/bubble.h"

#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <climits>

#include "utils/assert.h"
#include "core/tensor_builder.h"

namespace ts {
    const char *const Bubble::Parameter = "<param>";
    const char *const Bubble::Const = "<const>";
    const char *const Bubble::Variable = "<var>";
    static const std::unordered_set<std::string> EndPoints = {Bubble::Parameter, Bubble::Const, Bubble::Variable};

    std::string Bubble::RetentionParam::name = "#name";
    std::string Bubble::RetentionParam::op = "#op";
    std::string Bubble::RetentionParam::shape = "#shape";
    std::string Bubble::RetentionParam::dtype = "#dtype";

    const std::vector<std::string> &Bubble::RetentionParam::All() {
        static std::vector<std::string> all = {name, op, shape, dtype};
        return all;
    }

    bool Bubble::IsEndPoint(const std::string &op) {
        return EndPoints.find(op) != EndPoints.end();
    }

    Bubble::Bubble(self &&other) TS_NOEXCEPT {
        this->operator=(std::forward<self>(other));
    }

    Bubble::self &Bubble::operator=(self &&other) TS_NOEXCEPT {
        this->m_op = std::move(other.m_op);
        this->m_name = std::move(other.m_name);
        // this->m_output_count = other.m_output_count;
        this->m_params = std::move(other.m_params);
        return *this;
    }

    Bubble::Bubble(const std::string &op)
            : m_op(op) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name)
            : m_op(op), m_name(name) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const Shape &shape)
            : m_op(op), m_shape(shape) {
        update_retention_params();
    }

    Bubble::Bubble(const std::string &op, const std::string &name, const Shape &shape)
            : m_op(op), m_name(name), m_shape(shape) {
        update_retention_params();
    }

    bool Bubble::has(const std::string &param) const {
        return this->m_params.find(param) != this->m_params.end();
    }

    void Bubble::set(const std::string &param, const Tensor &value) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            this->m_params.insert(std::make_pair(param, value));
        } else {
            param_it->second = value;
        }
    }

    Tensor &Bubble::get(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        return param_it->second;
    }

    const Tensor &Bubble::get(const std::string &param) const {
        return const_cast<self *>(this)->get(param);
    }

    void Bubble::clear(const std::string &param) {
        auto param_it = m_params.find(param);
        if (param_it == m_params.end()) {
            TS_LOG_ERROR << "Unidentified param \"" << param << "\", did you mean \"" << fuzzy_param_name(param) << "\""
                         << eject;
        }
        this->m_params.erase(param_it);
    }

    void Bubble::clear_params() {
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

    std::string Bubble::fuzzy_param_name(const std::string &name) {
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

    void Bubble::update_retention_params() {
        TS_AUTO_CHECK(retention_param_sign == '#');
        set(RetentionParam::op, tensor::from(m_op));
        set(RetentionParam::name, tensor::from(m_name));
        // set(RetentionParam::output_count, tensor::from(m_output_count)); // not saving output count
        if (!m_shape.empty()) {
            set(RetentionParam::shape, tensor::from(m_shape));
        }
    }

    static size_t write_param(StreamWriter &stream, const std::pair<std::string, Tensor> &param) {
        auto &name = param.first;
        auto &value = param.second;
        size_t writen_size = 0;
        // 1 write param's name
        // 1.1 write name length
        writen_size += binio::write<uint32_t>(stream, uint32_t(name.size()));
        // 1.2 write name string
        writen_size += binio::write<char>(stream, name.data(), name.size());
        // 2. write param's value
        writen_size += value.serialize(stream);
        return writen_size;
    }

    static size_t read_param(StreamReader &stream, std::pair<std::string, Tensor> &param) {
        auto &name = param.first;
        auto &value = param.second;
        size_t read_size = 0;
        uint32_t size_buffer;
        // 1. read param's name
        // 1.1 read name length
        read_size += binio::read<uint32_t>(stream, size_buffer);
        // 1.2 read name
        std::vector<char> string_buffer(size_buffer);
        read_size += binio::read<char>(stream, string_buffer.data(), size_buffer);
        name = std::string(string_buffer.begin(), string_buffer.end());
        // 2. read param's value
        read_size += value.externalize(stream);
        return read_size;
    }

    size_t Bubble::serialize(StreamWriter &stream) const {
        size_t writen_size = 0;
        writen_size += binio::write<uint32_t>(stream, uint32_t(m_params.size()));
        for (auto &param : m_params) {
            writen_size += write_param(stream, param);
        }
        return writen_size;
    }

    size_t Bubble::externalize(StreamReader &stream) {
        m_params.clear();
        size_t read_size = 0;
        uint32_t size_buffer;
        read_size += binio::read<uint32_t>(stream, size_buffer);
        std::pair<std::string, Tensor> param;
        for (uint32_t i = 0; i < size_buffer; ++i) {
            read_size += read_param(stream, param);
            m_params.insert(param);
        }
        m_op = tensor::to_string(m_params[RetentionParam::op]);
        m_name = tensor::to_string(m_params[RetentionParam::name]);

        {
            auto it = m_params.find("#output_count");
            if (it != m_params.end()) {
                auto m_output_count = tensor::to_int(it->second);
                if (m_output_count != 1) {
                    TS_LOG_ERROR << "All operators' output count must be 1." << eject;
                }
            }
        }
        {
            auto it = m_params.find(RetentionParam::shape);
            if (it != m_params.end()) {
                auto tshape = tensor::cast(INT32, it->second);
                m_shape.resize(tshape.count());
                for (size_t i = 0; i < m_shape.size(); ++i) {
                    m_shape[i] = tshape.data<int32_t>(i);
                }
            }
        }
        return read_size;
    }

    const Shape Bubble::shape() const {
        return m_shape;
    }

    DTYPE Bubble::dtype() const {
        if (!has(RetentionParam::dtype)) return VOID;
        return DTYPE(tensor::to_int(get(RetentionParam::dtype)));
    }

    void Bubble::op(const std::string &_op) {
        m_op = _op;
        m_params[RetentionParam::op] = tensor::from(m_op);
    }

    void Bubble::name(const std::string &_name) {
        m_name = _name;
        m_params[RetentionParam::name] = tensor::from(m_name);
    }

    void Bubble::shape(const Shape &shape) {
        m_shape = shape;
        m_params[RetentionParam::shape] = tensor::from(m_shape);
    }

    void Bubble::dtype(DTYPE _dtype) {
        m_params[RetentionParam::dtype] = tensor::from<int32_t>(_dtype);
    }

    Bubble::Bubble(const std::string &op, int output_count)
            : self(op) {
        TS_AUTO_CHECK(output_count == 1);
    }

    Bubble::Bubble(const std::string &op, const std::string &name, int output_count)
            : self(op, name) {
        TS_AUTO_CHECK(output_count == 1);
    }

    Bubble::Bubble(const std::string &op, int output_count, const Shape &shape)
            : self(op, shape) {
        TS_AUTO_CHECK(output_count == 1);
    }

    Bubble::Bubble(const std::string &op, const std::string &name, int output_count, const Shape &shape)
            : self(op, name, shape) {
        TS_AUTO_CHECK(output_count == 1);
    }
}
