//
// Created by kier on 2019/2/18.
//

#include "backend/base/base_flatten.h"

#include <numeric>

#include "core/tensor_builder.h"
#include "backend/name.h"

namespace ts {
    namespace base {
        Flatten::Flatten() {
            field(name::dim, OPTIONAL, tensor::from<int32_t>(1));
        }

        void Flatten::init() {
            m_dim = tensor::to_int(get(name::dim));
            // TS_AUTO_CHECK(m_dim >= 0);
        }

        Shape Flatten::newshape(const Tensor &x) {
            auto fixed_dim = m_dim;
            if (fixed_dim < 0) fixed_dim += x.dims();
            auto need_size = size_t(fixed_dim + 1);
            auto x_size = x.sizes().size();
            if (need_size < x_size) {
                auto &size = x.sizes();
                std::vector<int> shape(size.begin(), size.begin() + need_size);
                shape.back() = std::accumulate(size.begin() + fixed_dim, size.end(), 1, std::multiplies<int>());
                return std::move(shape);
            } else if (need_size > x_size) {
                std::vector<int> ones(need_size - x_size, 1);
                auto shape = x.sizes();
                shape.insert(shape.end(), ones.begin(), ones.end());
                return std::move(shape);
            } else {
                return x.sizes();
            }
        }
    }
}
