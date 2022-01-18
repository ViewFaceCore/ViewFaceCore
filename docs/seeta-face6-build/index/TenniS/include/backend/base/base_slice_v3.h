//
// Created by kier on 2019-04-08.
//

#ifndef TENSORSTACK_BACKEND_BASE_SLICE_V3_H
#define TENSORSTACK_BACKEND_BASE_SLICE_V3_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class SliceV3 : public OperatorOnDevice {
        public:
            using self = SliceV3;
            using supper = OperatorOnDevice;

            SliceV3();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x origin input
             * @param begin begin
             * @param end end
             * @param stride stride
             * @param out output shape
             */
            virtual void slice(
                    const Tensor &x,
                    const std::vector<int> &begin,
                    const std::vector<int> &end,
                    const std::vector<int> &stride,
                    Tensor &out) = 0;

        private:
            void load_params(const Stack &stack);

            Shape m_begin;
            Shape m_end;
            Shape m_stride;

            int32_t m_begin_mask;
            int32_t m_end_mask;
            int32_t m_ellipsis_mask;
            int32_t m_new_axis_mask;
            int32_t m_shrink_axis_mask;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_SLICE_V3_H
