//
// Created by kier on 2019/2/17.
//

#include <backend/base/base_dimshuffle.h>

#include "backend/base/base_dimshuffle.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {

        Dimshuffle::Dimshuffle() {
            field(name::dim, REQUIRED);
            field(name::shuffle, REQUIRED);
        }

        void Dimshuffle::init() {
            supper::init();

            m_dim = tensor::to_int(get(name::dim));

            TS_AUTO_CHECK(m_dim >= 0);

            auto shuffle_tensor = tensor::cast(INT32, get(name::shuffle));

            TS_AUTO_CHECK(shuffle_tensor.dims() == 1);
            TS_AUTO_CHECK(shuffle_tensor.size(0) > 0);

            auto count = shuffle_tensor.count();
            m_shuffle.resize(size_t(count));
            for (int i = 0; i < count; ++i) {
                m_shuffle[i] = shuffle_tensor.data<int32_t>(size_t(i));
                TS_AUTO_CHECK(m_shuffle[i] >= 0);
            }
        }

        int Dimshuffle::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();
            TS_AUTO_CHECK(input_num == 1);

            auto x = stack[0];
            auto dim_size = x.size(m_dim);

            TS_AUTO_CHECK(m_dim < x.dims());

            for (auto &i : m_shuffle) {
                if (i >= dim_size) {
                    TS_LOG_ERROR << "Unsupported x.shape = " << to_string(x.sizes())
                    << ", dim = " << m_dim
                    << ", shuffle = " << to_string(m_shuffle) << "." << eject;
                }
            }

            auto output_shape = x.sizes();
            output_shape[m_dim] = int(m_shuffle.size());

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), output_shape);

            return 1;
        }

        int Dimshuffle::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;
            infer(stack, output);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto out = *stack.push(output[0], memory_device);

            dimshuffle(x, m_dim, m_shuffle, out);

            return 1;
        }
    }
}
