//
// Created by kier on 2019/2/20.
//

#include <backend/base/base_reshape.h>

#include "backend/base/base_reshape.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Reshape::Reshape() {
            field(name::shape, REQUIRED);
        }

        void Reshape::init() {
            supper::init();

            auto shape_tensor = tensor::cast(INT32, get(name::shape));

            TS_AUTO_CHECK(shape_tensor.dims() == 1);

            size_t count = size_t(shape_tensor.size(0));

            m_shape.clear();
            m_shape.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                m_shape.emplace_back(shape_tensor.data<int32_t>(i));
            }

            m_broadcast_dim = -1;
            m_count_without_dim = 1;

            for (size_t i = 0; i < count; ++i) {
                if (m_shape[i] == 0) continue;
                if (m_shape[i] > 0) {
                    m_count_without_dim *= m_shape[i];
                    continue;
                }
                if (m_broadcast_dim >= 0) TS_LOG_ERROR << "Can not reshape tensor to " << to_string(m_shape) << eject;
                m_broadcast_dim = int(i);
            }
        }

        Shape Reshape::newshape(const Tensor &x) {
            auto x_count = x.count();

            auto shape = m_shape;
            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == 0) {
                    if (i >= x.dims()) {
                        TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(m_shape) << eject;
                    }
                    shape[i] = x.size(i);
                }
            }

            if (m_broadcast_dim >= 0) {
                shape[m_broadcast_dim] = x_count / m_count_without_dim;
                if (shape[m_broadcast_dim] * m_count_without_dim != x_count) {
                    TS_LOG_ERROR << "Can not reshape " << to_string(x.sizes()) << " to " << to_string(m_shape) << eject;
                }
            }

            return shape;
        }
    }
}
