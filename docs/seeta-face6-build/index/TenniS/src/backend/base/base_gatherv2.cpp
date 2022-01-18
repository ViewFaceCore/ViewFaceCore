//
// Created by kier on 2019/3/6.
//

#include <backend/base/base_gatherv2.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        GatherV2::GatherV2() {
        }

        void GatherV2::init() {
            supper::init();

        }

        static Tensor::Prototype infer_gather(const Tensor &x, const Tensor &indices) {
            // auto dims = int(x.dims());

            TS_AUTO_CHECK(x.dims() >= 1);
            TS_AUTO_CHECK(indices.dims() >= 1);

            Shape output_shape = indices.sizes();
            output_shape.erase(output_shape.end() - 1);

            auto &indices_shape = indices.sizes();
            auto input_shape = x.sizes();
            TS_AUTO_CHECK(indices_shape[indices_shape.size() - 1] <= input_shape.size());
            output_shape.insert(output_shape.end(), input_shape.begin() + indices_shape[indices_shape.size() - 1], input_shape.end());

            return Tensor::Prototype(x.dtype(), std::move(output_shape));
        }

        int GatherV2::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto &x = stack[0];
            auto indices =  tensor::cast(INT32, stack[1]);

            output.resize(1);
            output[0] = infer_gather(x, indices);

            return 1;
        }

        int GatherV2::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);
            //auto indices =  stack[1].view(memory_device);//tensor::cast(INT32, stack[1]);
            //auto output_proto = infer_gather(stack[0], indices);

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            auto indices = stack[1].view(memory_device);
            TS_AUTO_CHECK(indices.dtype() == INT32);
            auto output_proto = infer_gather(stack[0], indices);

            auto &out = *stack.push(output_proto, memory_device);

            gather(x, indices, out);

            return 1;
        }
    }
}
