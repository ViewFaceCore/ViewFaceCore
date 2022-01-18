//
// Created by kier on 2019/7/23.
//

#include "backend/base/base_reduce_mean.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>
#include <iterator>

namespace ts {
    namespace base {
        ReduceMean::ReduceMean() {
            field("dims", REQUIRED);
            field("keep_dims", OPTIONAL, tensor::from<bool>(true));
        }

        void ReduceMean::init() {
            supper::init();

            m_dims.clear();
            auto dims_tensor = this->get("dims");
            int dims_dim = int(dims_tensor.dims());
            size_t dims_count = dims_dim == 0 ? 1 : (size_t)dims_tensor.size(0);
            m_dims.resize(dims_count);
            auto dims_data = dims_tensor.data<int>();
            for (size_t i = 0; i < dims_count; ++i) {
                m_dims[i] = dims_data[i];
            }

            m_keep_dim = tensor::to_bool(this->get("keep_dims"));
        }

        static bool check_dims_continue(std::vector<int> dims){
            bool continue_flag = true;
            for (int i = 0; i < dims.size() - 1; ++i) {
                if(dims[i + 1] != dims[i] + 1){
                    continue_flag = false;
                    break;
                }
            }
            return continue_flag;
        }

        static std::vector<int> checkout(Stack &stack, std::vector<int> dims, bool keep_dim, Shape &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            Shape input = stack[0].sizes();
            auto has_dims = int(input.size());

            bool continue_flag = true;
            if(dims.size() > 1){
                continue_flag = check_dims_continue(dims);
            }

            if(!continue_flag)
                TS_LOG_ERROR << "Dimensions must be continuous now!" << eject;

            std::vector<int> fixed_dims(dims);
            for (int i = 0; i < fixed_dims.size(); ++i) {
                fixed_dims[i] = dims[i] >= 0 ? dims[i] : has_dims + dims[i];
                if (fixed_dims[i] < 0 || fixed_dims[i] >= has_dims) {
                    TS_LOG_ERROR << "Reduce dim must in [-"
                                 << has_dims << ", "
                                 << has_dims << ")" << eject;
                }
            }

            if(keep_dim){
                for (int i = 0; i < fixed_dims.size(); ++i) {
                    input[fixed_dims[i]] = 1;
                }
                output = std::move(input);
            }
            else{
                int i = 0;
                for (int j = 0; i < input.size() && j < fixed_dims.size(); ++i) {
                    if(i == fixed_dims[j]){
                        ++j;
                    }
                    else{
                        output.emplace_back(input[i]);
                    }
                }
                std::copy(std::next(input.begin(),i),input.end(),std::back_inserter(output));
            }

            return fixed_dims;
        }

        int ReduceMean::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            Shape output_shape;
            TS_AUTO_CHECK(m_dims.size() >= 1);
            checkout(stack, m_dims, m_keep_dim, output_shape);

            output.resize(1);
            output[0] = Tensor::Prototype(stack[0].dtype(), output_shape);

            return 1;
        }

        int ReduceMean::run(Stack &stack) {
            Shape output_shape;
            auto fixed_dim = checkout(stack, m_dims, m_keep_dim, output_shape);

            Tensor::Prototype output_proto(stack[0].dtype(), output_shape);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto out = *stack.push(x.dtype(), output_shape, memory_device);

            reduce(x, fixed_dim, out);

//            if (!m_keep_dim) {
//                output_shape.erase(output_shape.begin() + fixed_dim);
//                auto fixed_out = out.reshape(output_shape);
//                stack.pop();
//                stack.push(fixed_out);
//            }

            return 1;
        }
    }
}
