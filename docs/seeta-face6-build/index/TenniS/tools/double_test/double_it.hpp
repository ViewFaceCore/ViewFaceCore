//
// Created by kier on 2019/3/25.
//

#ifndef TENSORSTACK_DOUBLE_IT_H
#define TENSORSTACK_DOUBLE_IT_H

#include "../run_test/test_case.hpp"

#include "backend/name.h"

namespace ts {
    class BatchMark {
    public:
        struct {
            std::vector<bool> input;
            std::vector<bool> output;
        } batch;
        struct {
            std::vector<bool> input;
            // std::vector<bool> output;
        } shuffle;
    };

    inline BatchMark default_mark(size_t input_count, size_t output_count) {
        BatchMark mark;
        mark.batch.input.resize(input_count, false);
        mark.batch.output.resize(output_count, true);
        mark.shuffle.input.resize(input_count, false);
        if (input_count) mark.batch.input[0] = true;
        if (input_count) mark.shuffle.input[0] = true;

        return std::move(mark);
    }

    inline BatchMark get_mark(const TestCase &tc) {
        auto mark = default_mark(tc.input_count, tc.output_count);
        if (tc.op == name::layer::shape_index_patch()) {
            mark.batch.input[1] = true;
        } else if (
                tc.op == name::layer::gather() ||
                tc.op == name::layer::shape() ||
                tc.op == name::layer::reshape() ||
                tc.op == name::layer::reshape_v2() ||
                tc.op == name::layer::concat() ||
                false) {
            for (auto it = mark.batch.input.begin(); it != mark.batch.input.end(); ++it) {
                *it = false;
            }
            for (auto it = mark.batch.output.begin(); it != mark.batch.output.end(); ++it) {
                *it = false;
            }
            for (auto it = mark.shuffle.input.begin(); it != mark.shuffle.input.end(); ++it) {
                *it = false;
            }
        } else if (
                tc.op == name::layer::onnx_pooling2d_padding() ||
                tc.op == name::layer::mx_pooling2d_padding() ||
                tc.op == name::layer::tf_conv2d_padding() ||
                tc.op == name::layer::tf_pooling2d_padding() ||
                false) {
                for (auto it = mark.batch.output.begin(); it != mark.batch.output.end(); ++it) {
                    *it = false;
                }
        } else if (tc.input_count > 0 && tc.input.at(0).dims() == 0) {
            for (auto it = mark.batch.input.begin(); it != mark.batch.input.end(); ++it) {
                *it = false;
            }
            for (auto it = mark.batch.output.begin(); it != mark.batch.output.end(); ++it) {
                *it = false;
            }
            for (auto it = mark.shuffle.input.begin(); it != mark.shuffle.input.end(); ++it) {
                *it = false;
            }
        }
        return mark;
    }

    inline void swap(void *lhs, void *rhs, int width) {
        auto lhs_data = reinterpret_cast<char *>(lhs);
        auto rhs_data = reinterpret_cast<char *>(rhs);
        for (int i = 0; i < width; ++i) {
            std::swap(*lhs_data, *rhs_data);
            ++lhs_data;
            ++rhs_data;
        }
    }

    inline void inverse(void *data, int count, int width) {
        auto data_char = reinterpret_cast<char *>(data);
        auto half = count / 2;
        for (int i = 0; i < half; ++i) {
            auto lhs = &data_char[i * width];
            auto rhs = &data_char[(count - 1 - i) * width];
            swap(lhs, rhs, width);
        }
    }

    /**
     * generate sister test case, by reverse test cases
     * @param tc test case
     * @return sister test case
     */
    inline TestCase sister(const TestCase &tc) {
        Workbench bench(ComputingDevice(CPU, 0));

        auto batch_mark = get_mark(tc);

        TestCase tc2 = tc;

        for (size_t i = 0; i < batch_mark.shuffle.input.size(); ++i) {
            auto shuffle = batch_mark.shuffle.input[i];
            if (!shuffle) continue;

            auto input = tc2.input[i].clone();

            inverse(input.data(), input.count(), input.proto().type_bytes());

            tc2.input[i] = input;
        }


        Bubble bubble(tc2.op, tc2.op, tc2.output_count);
        for (auto &param_pair: tc2.param) {
            bubble.set(param_pair.first, param_pair.second);
        }

        Operator::shared built_op = bench.offline_create(bubble, true);

        std::vector<Tensor> input_vector(tc2.input_count);
        std::vector<Tensor> output_vector(tc2.output_count);

        for (auto &input_pair : tc2.input) {
            input_vector[input_pair.first] = input_pair.second;
        }

        for (auto &output_pair : tc2.output) {
            output_vector[output_pair.first] = output_pair.second;
        }

        std::vector<Tensor> run_output;
        bench.offline_run(built_op, input_vector, run_output);

        for (int i = 0; i < run_output.size(); ++i) {
            tc2.output[i] = run_output[i];
        }

        return tc2;
    }

    static inline TestCase concat(const TestCase &tc1, const TestCase &tc2) {
        assert(tc1.op == tc2.op);
        auto batch_mark = get_mark(tc1);
        Workbench bench(ComputingDevice(CPU, 0));

        auto tc3 = tc1;

        Operator::shared concat_op = OperatorCreator::Create(CPU, name::layer::concat(), true);
        concat_op->set(name::dim, tensor::from(0));
        concat_op->init();
        auto &stack = bench.stack();
        // concat input
        for (int i = 0; i < tc1.input_count; ++i) {
            auto batch = batch_mark.batch.input[i];
            if (!batch) continue;
            stack.push(tc1.input.at(i));
            stack.push(tc2.input.at(i));
            RunOperator(concat_op, stack, 2);
            tc3.input[i] = stack[-1].clone();
            stack.clear();
        }
        // concat output
        for (int i = 0; i < tc1.output_count; ++i) {
            auto batch = batch_mark.batch.output[i];
            if (!batch) continue;
            stack.push(tc1.output.at(i));
            stack.push(tc2.output.at(i));
            RunOperator(concat_op, stack, 2);
            tc3.output[i] = stack[-1].clone();
            stack.clear();
        }
        return tc3;
    }
}

#endif //TENSORSTACK_DOUBLE_IT_H
