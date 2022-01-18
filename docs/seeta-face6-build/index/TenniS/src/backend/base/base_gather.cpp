//
// Created by kier on 2019/3/6.
//

#include <backend/base/base_gather.h>

#include "backend/base/base_gather.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Gather::Gather() {
            field(name::axis, OPTIONAL, tensor::from<int32_t>(0));
        }

        void Gather::init() {
            supper::init();

            m_axis = tensor::to_int(get(name::axis));
        }

        static Tensor::Prototype infer_gather(const Tensor &x, const Tensor &indices, int axis) {
            auto dims = int(x.dims());

            TS_AUTO_CHECK(x.dims() >= 1);
            TS_AUTO_CHECK(axis >= -dims && axis < dims);

            if (axis < 0) axis = int(x.dims()) + axis;

            auto output_shape = x.sizes();

            output_shape.erase(output_shape.begin() + axis);

            auto &indices_shape = indices.sizes();

            output_shape.insert(output_shape.begin() + axis, indices_shape.begin(), indices_shape.end());

            return Tensor::Prototype(x.dtype(), std::move(output_shape));
        }

        int Gather::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto indices =  tensor::cast(INT32, stack[1]);

            output.resize(1);
            output[0] = infer_gather(x, indices, m_axis);

            return 1;
        }

        int Gather::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto indices =  tensor::cast(INT32, stack[1]);
            auto output_proto = infer_gather(stack[0], indices, m_axis);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);
            auto &out = *stack.push(output_proto, memory_device);

            auto positive_axis = m_axis >= 0 ? m_axis : int(x.dims()) + m_axis;

            gather(x, indices, positive_axis, out);

            return 1;
        }
    }
}
