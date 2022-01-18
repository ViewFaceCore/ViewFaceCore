//
// Created by kier on 2019/9/3.
//

#include "backend/base/base_roi_align.h"
#include <core/tensor_builder.h>

namespace ts {
    namespace base {
        ROIAlign::ROIAlign() {
            field("pool_h", REQUIRED);
            field("pool_w", REQUIRED);
            field("spatial_scale", OPTIONAL, tensor::from<float>(1.0f));
            field("sampling_ratio", OPTIONAL, tensor::from<int32_t>(2));
        }

        void ROIAlign::init() {
            supper::init();

            m_pool_h = tensor::to_int(get("pool_h"));
            m_pool_w = tensor::to_int(get("pool_w"));
            m_spatial_scale = tensor::to_float(get("spatial_scale"));
            m_sampling_ratio = tensor::to_int(get("sampling_ratio"));
        }

        int ROIAlign::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_ASSERT(stack.size() == 2);
            TS_AUTO_ASSERT(stack[0].dtype() == stack[1].dtype());

            output.resize(1);
            output[0] = Tensor::Prototype(stack[0].dtype(), {stack[1].size(0), stack[0].size(1), m_pool_h, m_pool_w});

            return 1;
        }

        int ROIAlign::run(Stack &stack) {
            TS_AUTO_ASSERT(stack.size() == 2);
            TS_AUTO_ASSERT(stack[0].dtype() == stack[1].dtype());

            auto memory_device = this->running_memory_device();

            std::vector<Tensor> inputs;
            for (size_t i = 0; i < stack.size(); ++i) {
                inputs.emplace_back(stack[i].view(memory_device));
            }

            auto outputs = roi_align(inputs, m_pool_h, m_pool_w, m_spatial_scale, m_sampling_ratio);

            Tensor packed;
            packed.pack(outputs);

            stack.push(packed);

            return 1;
        }
    }
}