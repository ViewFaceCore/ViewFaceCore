//
// Created by kier on 2019/2/15.
//

#include <backend/base/base_concat.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        Concat::Concat() {
            field(name::dim, REQUIRED);
        }

        void Concat::init() {
            supper::init();

            m_dim = tensor::to_int(this->get(name::dim));

            // TS_AUTO_CHECK(m_dim >= 0);
        }

        static void throw_error_message(const std::string &title, Stack &stack, int dim) {
            auto N = int(stack.size());
            std::ostringstream oss;
            oss << "{";
            for (int j = 0; j < N; ++j) {
                if (j) oss << ", ";
                oss << stack[j].proto();
            }
            oss << "}";
            TS_LOG_ERROR << title << "Can not concat " << oss.str() << " at dim=" << dim << ts::eject;
        }

        int Concat::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();

            TS_AUTO_CHECK(input_num != 0);
            // TS_AUTO_CHECK(stack.index(0)->dtype() == FLOAT32 || stack.index(0)->dtype() == FLOAT64);

            if (input_num == 1)
            {
                output.resize(1);
                output[0] = stack[0].proto();
                return 1;
            }

            auto dtype = stack.index(0)->dtype();

            for (int i = 1; i < input_num; i++)
            {
                if (stack.index(i)->dtype() != dtype) {
                    std::ostringstream oss;
                    oss << "(";
                    for (int j = 0; j < input_num; ++j) {
                        if (j) oss << ", ";
                        oss << type_str(stack.index(j)->dtype());
                    }
                    oss << ")";
                    TS_LOG_ERROR << "Can not concat " << oss.str() << ts::eject;
                }
            }

            Shape output_shape(stack.index(0)->sizes());
            int output_dims = int(output_shape.size());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            if (fixed_dim < 0 || fixed_dim >= output_dims) {
                TS_LOG_ERROR << "Concat dim must in [-"
                    << output_dims << ", "
                    << output_dims << ")" << eject;
            }

            auto num_dims = output_shape.size();
            int concat_dim_output_num = output_shape[fixed_dim];

            for (size_t i = 1; i < input_num; i++)
            {
                auto shape = stack.index(int(i))->sizes();
                if(shape.size() != num_dims) {
                    throw_error_message("", stack, m_dim);
                }

                for (int j = 0; j < shape.size(); j++)
                {
                    if (j == fixed_dim)
                        continue;
                    if(shape[j] != output_shape[j]) {
                        throw_error_message("", stack, m_dim);
                    }
                }
                concat_dim_output_num += shape[fixed_dim];
            }

            output_shape[fixed_dim] = concat_dim_output_num;

            output.resize(1);
            output[0] = Tensor::Prototype(dtype, output_shape);

            return 1;
        }

        int Concat::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            infer(stack, output_protos);

            auto input_num = stack.size();

            auto memory_device = running_memory_device();

            std::vector<Tensor> x;
            for (size_t i = 0; i < input_num; ++i) {
                x.emplace_back(stack[i].view(memory_device));
            }

            Tensor out = *stack.push(output_protos[0], memory_device);

            int output_dims = int(x[0].dims());
            int fixed_dim = m_dim >= 0 ? m_dim : output_dims + m_dim;

            if (fixed_dim < 0 || fixed_dim >= output_dims) {
                TS_LOG_ERROR << "Concat dim must in [-"
                             << output_dims << ", "
                             << output_dims << ")" << eject;
            }

            concat(x, fixed_dim, out);

            return 1;
        }
    }
}
