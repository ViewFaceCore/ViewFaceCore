//
// Created by kier on 2019/3/14.
//

#include "backend/base/base_shape_index_patch.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        const std::string ShapeIndexPatch::param_origin_patch = "origin_patch";
        const std::string ShapeIndexPatch::param_origin = "origin";

        ShapeIndexPatch::ShapeIndexPatch() {
            field(param_origin_patch, REQUIRED);
            field(param_origin, REQUIRED);
        }

        void ShapeIndexPatch::init() {
            supper::init();

            auto &tensor_origin_patch = get(param_origin_patch);
            auto &tensor_origin = get(param_origin);

            TS_AUTO_CHECK(tensor_origin_patch.has_shape(2) && tensor_origin.has_shape(2));

            auto int32_origin_path = tensor::cast(INT32, tensor_origin_patch);
            auto int32_origin = tensor::cast(INT32, tensor_origin);

            m_origin_patch.height = int32_origin_path.data<int32_t>(0);
            m_origin_patch.width = int32_origin_path.data<int32_t>(1);
            m_origin.height = int32_origin.data<int32_t>(0);
            m_origin.width = int32_origin.data<int32_t>(1);
        }

        int ShapeIndexPatch::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            auto input_num = stack.size();
            TS_AUTO_CHECK(input_num == 2);

            auto &x = stack[0];
            auto &pos = stack[1];

            TS_AUTO_CHECK(x.dims() == 4);
            TS_AUTO_CHECK(pos.dims() == 4);
            TS_AUTO_CHECK(x.dtype() == pos.dtype());

            auto landmarkx2 = pos.size(1);

            TS_AUTO_CHECK(landmarkx2 % 2 == 0);

            auto &x_shape = x.sizes();

            int x_patch_h = int(m_origin_patch.height * x_shape[2] / float(m_origin.height) + 0.5f);
            int x_patch_w = int(m_origin_patch.width * x_shape[3] / float(m_origin.width) + 0.5f);

            std::vector<int> out_shape = {x_shape[0], x_shape[1], x_patch_h, landmarkx2 / 2, x_patch_w};

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), std::move(out_shape));

            return 1;
        }

        int ShapeIndexPatch::run(Stack &stack) {
            std::vector<Tensor::Prototype> output;

            infer(stack, output);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);
            auto pos = stack[1].view(memory_device);

            auto &out = *stack.push(output[0], memory_device);

            sample(x, pos, m_origin_patch, m_origin, out);

            return 1;
        }
    }
}