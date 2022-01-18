//
// Created by kier on 2019/3/5.
//

#include "backend/base/base_unsqueeze.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Unsqueeze::Unsqueeze() {
            field(name::axes, REQUIRED);
        }

        void Unsqueeze::init() {
            supper::init();

            auto axes_tensor = tensor::cast(INT32, get(name::axes));

            TS_AUTO_CHECK(axes_tensor.dims() == 1 || axes_tensor.dims() == 0);

            size_t count = size_t(axes_tensor.count());

            m_axes.clear();
            m_axes.reserve(count);

            for (size_t i = 0; i < count; ++i) {
                m_axes.emplace_back(axes_tensor.data<int32_t>(i));
            }
        }

        Shape Unsqueeze::newshape(const Tensor &x) {
            auto shape = x.sizes();

            for (auto axis : m_axes) {
                auto max_axis = int32_t(shape.size());
                if (axis > max_axis || axis < -max_axis) {
                    TS_LOG_ERROR << op() << " do not support unsqueeze shape=" << to_string(x.sizes())
                                 << " with axes=" << to_string(m_axes) << eject;
                }
                if (axis >= 0) {
                    shape.insert(shape.begin() + axis, 1);
                } else {
                    shape.insert(shape.end() + axis + 1, 1);
                }
            }

            return shape;
        }
    }
}
