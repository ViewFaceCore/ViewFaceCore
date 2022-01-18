//
// Created by kier on 2019/9/3.
//

#include "backend/base/base_proposal.h"
#include <core/tensor_builder.h>

namespace ts {
    namespace base {
        Proposal::Proposal() {
            field("strides", REQUIRED);
            field("ratios", REQUIRED);
            field("scales", REQUIRED);
            field("pre_nms_top_n", OPTIONAL, tensor::from<int32_t>(6000));
            field("post_nms_top_n", OPTIONAL, tensor::from<int32_t>(300));
            field("nms_thresh", OPTIONAL, tensor::from<float>(0.7f));
            field("min_size", OPTIONAL, tensor::from<int32_t>(16));
            field("min_level", OPTIONAL, tensor::from<int32_t>(2));
            field("max_level", OPTIONAL, tensor::from<int32_t>(5));
            field("canonical_scale", OPTIONAL, tensor::from<int32_t>(224));
            field("canonical_level", OPTIONAL, tensor::from<int32_t>(4));
        }

        void Proposal::init() {
            supper::init();

            m_strides = tensor::array::to_int(get("strides"));
            m_ratios = tensor::array::to_float(get("ratios"));
            m_scales = tensor::array::to_float(get("scales"));
            m_pre_nms_top_n = tensor::to_int(get("pre_nms_top_n"));
            m_post_nms_top_n = tensor::to_int(get("post_nms_top_n"));
            m_nms_thresh = tensor::to_float(get("nms_thresh"));
            m_min_size = tensor::to_int(get("min_size"));
            m_min_level = tensor::to_int(get("min_level"));
            m_max_level = tensor::to_int(get("max_level"));
            m_canonical_scale = tensor::to_int(get("canonical_scale"));
            m_canonical_level = tensor::to_int(get("canonical_level"));
        }

        int Proposal::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_ASSERT(stack.size() >= 3);

            auto num_images = stack[0].size(0);

            auto output_size = m_max_level - m_min_level + 1;

            output.resize(output_size);
            for (auto &proto : output) {
                proto = Tensor::Prototype(stack[-3].dtype(), {num_images * m_post_nms_top_n, 5});
            }

            return 1;
        }

        int Proposal::run(Stack &stack) {
            TS_AUTO_ASSERT(stack.size() >= 3);
            
            auto memory_device = this->running_memory_device();

            std::vector<Tensor> inputs;
            for (size_t i = 0; i < stack.size(); ++i) {
                inputs.emplace_back(stack[i].view(memory_device));
            }

            auto outputs = proposal(inputs,
                    m_strides, m_ratios, m_scales,
                    m_pre_nms_top_n, m_post_nms_top_n, m_nms_thresh,
                    m_min_size, m_min_level, m_max_level,
                    m_canonical_scale, m_canonical_level);

            Tensor packed;
            packed.pack(outputs);

            stack.push(packed);

            return 1;
        }
    }
}