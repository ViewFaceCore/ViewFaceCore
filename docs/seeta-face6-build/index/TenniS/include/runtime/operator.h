//
// Created by kier on 2018/5/19.
//

#ifndef TENSORSTACK_RUNTIME_OPERATOR_H
#define TENSORSTACK_RUNTIME_OPERATOR_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <core/tensor.h>

namespace ts {
    class Stack;

    class TS_DEBUG_API Operator {
    public:
        using self = Operator;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        template<typename K, typename V>
        using hash_map = std::unordered_map<K, V>;

        template<typename K>
        using hash_set = std::unordered_set<K>;

        const static char retention_param_sign = '#';

        enum FieldAttr {
            OPTIONAL,
            REQUIRED,
        };

        virtual ~Operator() = default;

        // explicit Operator(const std::string &name) : m_name(name) {}
        /**
         * @note bind **DeviceContext** before.
         */
        virtual void init();

        // work on tensor stack, same as call
        /**
         * @param stack running stack
         * @return output param
         * @note bind **DeviceContext** before
         * @note must return 1
         */
        virtual int run(Stack &stack) = 0;

        // infer output size by input size, only for size allocate
        /**
         *
         * @param stack running stack
         * @param output output tensor's proto
         * @return keep value
         * @note bind ThreadPool, DeviceContext, RuntimeContext before
         * @note operator's output count must be 1, this function get each field proto
         */
        virtual int infer(Stack &stack, std::vector<Tensor::Prototype> &output) = 0;

        /**
         * all operator must return an tensor or packed tensor
         * @param stack
         * @return tensor prototypes
         */
        TensorPrototype infer(Stack &stack);

        bool has(const std::string &param) const;

        // the params start with "#" are Retention parameters
        // now Retention parameters has #op, #name, #output_size
        void set(const std::string &param, const Tensor &value);

        Tensor &get(const std::string &param);

        const Tensor &get(const std::string &param) const;

        void clear(const std::string &param);

        void clear_params();

        /**
         * check if all required fields have been set
         * @return return true only if all required fields set
         */
        bool check_params() const;

        std::vector<std::string> unsatisfied_fields() const;

        void field(const std::string &param, FieldAttr attr, const Tensor &default_value);

        void field(const std::string &param, FieldAttr attr);

        void clear_fields();

        MemoryDevice memory_device() const;

        ComputingDevice computing_device() const;

        const hash_map<std::string, Tensor> &params() const;

        std::string op() const;

        std::string name() const;

        /**
         * @deprecated
         * @return 1
         */
        int output_count() const;

        enum class ParamCheckingMode : int32_t {
            WEAK = 0,
            STRICT = 1,
        };

        void set_param_checking_mode(ParamCheckingMode mode);

        std::vector<std::string> list_all_fields() const;

        std::vector<std::string> list_required_fields() const;

        std::vector<std::string> list_optional_fields() const;

    private:
        bool is_in_fields(const std::string &name);

        bool is_in_params(const std::string &name);

        std::string fuzzy_field_name(const std::string &name);

        std::string fuzzy_param_name(const std::string &name);

        hash_map<std::string, Tensor> m_params;
        hash_set<std::string> m_optional_fields;
        hash_set<std::string> m_required_fields;

        ParamCheckingMode m_param_checking_mode = ParamCheckingMode::STRICT;
    };

    /**
     * Run operator directly
     * @param op operator ready to run
     * @param stack the stack running on
     * @param nargs the number of input arguments
     * @return the number of return values
     * @note all input argument would be pop after running
     */
    TS_DEBUG_API int RunOperator(Operator::shared op, Stack &stack, int nargs);

    /**
     * Infer operator directly
     * @param op operator ready to run
     * @param stack the stack running on
     * @param nargs the number of input arguments
     * @return the number of return values
     * @note all input argument would be pop after running
     */
    TS_DEBUG_API int InferOperator(Operator::shared op, Stack &stack, int nargs, std::vector<Tensor::Prototype> &output);
}


#endif //TENSORSTACK_RUNTIME_OPERATOR_H
