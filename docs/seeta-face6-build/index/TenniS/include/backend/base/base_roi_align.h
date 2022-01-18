//
// Created by kier on 2019/9/3.
//

#ifndef TENSORSTACK_BASE_ROI_ALIGN_H
#define TENSORSTACK_BASE_ROI_ALIGN_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class ROIAlign : public OperatorOnDevice {
        public:
            using self = ROIAlign;
            using supper = OperatorOnDevice;

            ROIAlign();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual std::vector<Tensor> roi_align(
                    const std::vector<Tensor> &inputs,
                    int pool_h, int pool_w, float spatial_scale, int sampling_ratio) = 0;
        private:
            int m_pool_h = 0;
            int m_pool_w = 0;
            float m_spatial_scale = 1.0f;
            int m_sampling_ratio = 2;
        };
    }
}

#endif //TENSORSTACK_BASE_ROI_ALIGN_H
