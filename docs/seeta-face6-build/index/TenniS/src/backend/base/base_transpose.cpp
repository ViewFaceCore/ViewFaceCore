//
// Created by kier on 2019/2/21.
//

#include "backend/base/base_transpose.h"
#include <backend/name.h>
#include <backend/base/base_transpose.h>


#include "core/tensor_builder.h"
#include <algorithm>

namespace ts {
    namespace base {
        Transpose::Transpose() {
            field(name::permute, OPTIONAL);
        }

        void Transpose::init() {
            supper::init();

            m_permute.clear();
            if (has(name::permute)) {
                auto permute_tensor = tensor::cast(INT32, get(name::permute));

                TS_AUTO_CHECK(permute_tensor.dims() == 1);

                auto count = size_t(permute_tensor.size(0));
                auto permute_data = permute_tensor.data<int32_t>();

                m_permute.reserve(count);

                for (size_t i = 0; i < count; ++i) {
                    m_permute.emplace_back(permute_data[i]);
                }

                // check permute
                std::vector<bool> flag(m_permute.size(), false);
                for (auto &dim : m_permute) {
                    if (dim < 0 || dim >= m_permute.size()) break;
                    flag[dim] = true;
                }
                if (std::find(flag.begin(), flag.end(), false) != flag.end()) {
                    TS_LOG_ERROR << "Can not transpose to " << to_string(m_permute) << eject;
                }
            }
        }

        std::vector<int> Transpose::get_permute(const Tensor &x) {
            if (!m_permute.empty()) {
                if(x.dims() > m_permute.size()) {
                    TS_LOG_ERROR << "Can not transpose " << x.proto() << " with permute=" << to_string(m_permute) << eject;
                }
                if (x.dims() < m_permute.size()) {
                    // It's OK now!
                }
                return m_permute;
            }
            std::vector<int> permute(x.dims());
            for (size_t i = 0; i < permute.size(); ++i) {
                permute[i] = int(i);
            }
            if (permute.size() >= 2) {
                permute[permute.size() - 2] = permute[permute.size() - 1];
                permute[permute.size() - 1] = permute[permute.size() - 2];
            }
            return std::move(permute);
        }

        Shape permuted_shape(const Shape &shape, const std::vector<int> &permute) {
            auto fixed_shape = shape;
            while (fixed_shape.size()  < permute.size()) {
                fixed_shape.insert(fixed_shape.begin(), 1);
            }
            Shape newshape(fixed_shape.size());
            for (size_t i = 0; i < permute.size(); ++i) {
                newshape[i] = fixed_shape[permute[i]];
                TS_AUTO_CHECK(newshape[i] > 0);
            }
            return std::move(newshape);
        }

        int Transpose::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), permuted_shape(x.sizes(), get_permute(x)));

            return 1;
        }

        int Transpose::run(ts::Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto permute = get_permute(x);

            if (permute.size() > x.dims()) {
                auto fixed_shape = x.sizes();
                while (fixed_shape.size() < permute.size()) {
                    fixed_shape.insert(fixed_shape.begin(), 1);
                }
                x = x.reshape(fixed_shape);
            }

            Tensor::Prototype output_proto(x.dtype(), permuted_shape(x.sizes(), permute));

            auto out = *stack.push(output_proto, memory_device);

            transpose(x, permute, out);

            return 1;
        }
    }
}

