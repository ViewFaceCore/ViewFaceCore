//
// Created by kier on 2019/1/23.
//

#ifndef TENSORSTACK_MODULE_BUBBLE_H
#define TENSORSTACK_MODULE_BUBBLE_H


#include <memory>
#include <unordered_map>
#include "core/tensor.h"

#include "core/tensor_builder.h"

namespace ts {
    class TS_DEBUG_API Bubble : public Serializable {
    public:
        using self = Bubble;

        template<typename K, typename V>
        using map = std::unordered_map<K, V>;
        using param_dict = map<std::string, Tensor>;

        const static char retention_param_sign = '#';

        class RetentionParam {
        public:
            static std::string name;
            static std::string op;
            static std::string shape;
            static std::string dtype;

            static const std::vector<std::string> &All();
        };

        explicit Bubble() = default;

        Bubble(const self &) = default;

        self &operator=(const self &) = default;

        Bubble(self &&other) TS_NOEXCEPT;

        self &operator=(self &&other) TS_NOEXCEPT;

        explicit Bubble(
                const std::string &op);

        explicit Bubble(
                const std::string &op,
                const std::string &name);

        explicit Bubble(
                const std::string &op, const Shape &shape);

        explicit Bubble(
                const std::string &op,
                const std::string &name, const Shape &shape);

        explicit Bubble(
                const std::string &op, int output_count);

        explicit Bubble(
                const std::string &op,
                const std::string &name, int output_count);

        explicit Bubble(
                const std::string &op, int output_count, const Shape &shape);

        explicit Bubble(
                const std::string &op,
                const std::string &name, int output_count, const Shape &shape);

        // output
        size_t output_count() const { return 1; }

        const std::string &op() const { return m_op; }

        const std::string &name() const { return m_name; }

        const param_dict &params() const { return m_params; }

        bool has(const std::string &param) const;

        void set(const std::string &param, const Tensor &value);

        Tensor &get(const std::string &param);

        const Tensor &get(const std::string &param) const;

        void clear(const std::string &param);

        void clear_params();

        size_t serialize(StreamWriter &stream) const final;

        size_t externalize(StreamReader &stream) final;

        const Shape shape() const;

        DTYPE dtype() const;

        void op(const std::string &_op);

        void name(const std::string &_name);

        void shape(const Shape &shape);

        void dtype(DTYPE _dtype);

        int get_int(const std::string &param) const { return tensor::to_int(get(param)); }

        unsigned int get_uint(const std::string &param) const { return tensor::to_uint(get(param)); }

        float get_float(const std::string &param) const { return tensor::to_float(get(param)); }

        double get_double(const std::string &param) const { return tensor::to_double(get(param)); }

        std::string get_string(const std::string &param) const { return tensor::to_string(get(param)); }

        bool get_bool(const std::string &param) const { return tensor::to_bool(get(param)); }

        std::vector<int32_t> get_int_list(const std::string &param) const {
            return tensor::array::to_int(get(param));
        }

        std::vector<uint32_t> get_uint_list(const std::string &param) const {
            return tensor::array::to_uint(get(param));
        }

        std::vector<float> get_float_list(const std::string &param) const {
            return tensor::array::to_float(get(param));
        }

        std::vector<double> get_double_list(const std::string &param) const {
            return tensor::array::to_double(get(param));
        }

        std::vector<bool> get_bool_list(const std::string &param) const {
            return tensor::array::to_bool(get(param));
        }

    private:
        void update_retention_params();

        /**
         * Operator name
         */
        std::string m_op;
        /**
         * Bubble name
         */
        std::string m_name;

        /**
         * Parameters
         */
        param_dict m_params;

        /**
         * datum shape
         */
        Shape m_shape;

        /**
         *
         * @param name
         * @return
         */
        std::string fuzzy_param_name(const std::string &name);

    public:
        static const char *const Parameter; // Mean must input in graph
        static const char *const Const;     // Mean never change in graph, including weights, saving in static memory
        static const char *const Variable;  // Mean may change in graph, saving some state, do no use in this version

        static bool IsEndPoint(const std::string &op);
    };

    inline std::ostream &operator<<(std::ostream &out, const Bubble &op) {
        std::ostringstream oss;
        oss << "{op=\"" << op.op() << "\", name=\"" << op.name() << "\", out=" << op.output_count() << "}";
        return out << oss.str();
    }
}


#endif //TENSORSTACK_MODULE_BUBBLE_H
