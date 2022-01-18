//
// Created by kier on 2019/3/5.
//

#include "backend/base/base_reshape_v2.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        void ReshapeV2::init() {
            supper::init();
        }
        static inline Shape newshape(const Tensor &x, const Tensor &shape) {
            auto shape_tensor = tensor::cast(INT32, shape);

            TS_AUTO_CHECK(shape_tensor.dims() == 1);

            auto count = size_t(shape_tensor.size(0));

            Shape m_shape(count);

            for (size_t i = 0; i < count; ++i) {
                m_shape[i] = shape_tensor.data<int32_t>(i);
                if (m_shape[i] == 0 && i < x.dims()) {
                    m_shape[i] = x.size(i);
                }
            }

            int m_broadcast_dim = -1;
            int m_count_without_dim = 1;

            for (size_t i = 0; i < count; ++i) {
                if (m_shape[i] == 0)  TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(m_shape) << eject;
                if (m_shape[i] > 0) {
                    m_count_without_dim *= m_shape[i];
                    continue;
                }
                if (m_broadcast_dim >= 0)  TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(m_shape) << eject;
                m_broadcast_dim = int(i);
            }

            auto x_count = x.count();

            if (m_broadcast_dim >= 0) {
                m_shape[m_broadcast_dim] = x_count / m_count_without_dim;
                if (m_shape[m_broadcast_dim] * m_count_without_dim != x_count) {
                    TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(m_shape) << eject;
                }
            }

            return m_shape;
        }


        int ReshapeV2::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);


            auto &x = stack[0];

            stack.push(x.reshape(output[0].sizes()));

            return 1;
        }

        int ReshapeV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();
            TS_AUTO_CHECK(input_num == 2);

            auto &x = stack[0];
            auto &shape = stack[1];

            auto new_shape = newshape(x, shape);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), new_shape);

            auto &reshape_x = output[0];

            TS_AUTO_CHECK(x.count() == reshape_x.count());

            return 1;
        }
    }
}
