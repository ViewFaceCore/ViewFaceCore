//
// Created by kier on 2019/9/3.
//

#ifndef TENSORSTACK_BASE_PROPOSAL_H
#define TENSORSTACK_BASE_PROPOSAL_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Proposal : public OperatorOnDevice {
        public:
            using self = Proposal;
            using supper = OperatorOnDevice;

            Proposal();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual std::vector<Tensor> proposal(
                    const std::vector<Tensor> &inputs,
                    const std::vector<int32_t> &strides,
                    const std::vector<float> &ratios,
                    const std::vector<float> &scales,
                    int32_t pre_nms_top_n = 6000,
                    int32_t post_nms_top_n = 300,
                    float nms_thresh = 0.7f,
                    int32_t min_size = 16,
                    int32_t min_level = 2,
                    int32_t max_level = 5,
                    int32_t canonical_scale = 224,
                    int32_t canonical_level = 4) = 0;

        private:
            std::vector<int32_t> m_strides;
            std::vector<float> m_ratios;
            std::vector<float> m_scales;
            int32_t m_pre_nms_top_n = 6000;
            int32_t m_post_nms_top_n = 300;
            float m_nms_thresh = 0.7f;
            int32_t m_min_size = 16;
            int32_t m_min_level = 2;
            int32_t m_max_level = 5;
            int32_t m_canonical_scale = 224;
            int32_t m_canonical_level = 4;
        };
    }
}

#endif //TENSORSTACK_BASE_PROPOSAL_H
