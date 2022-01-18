//
// Created by kier on 19-4-17.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_DCN_V2_FORWARD_H
#define TENSORSTACK_BACKEND_BASE_BASE_DCN_V2_FORWARD_H

#include "backend/base/operator_on_device.h"
#include <valarray>

#include "backend/common_structure.h"


namespace ts {
    namespace base {
        class DCNV2Forward : public OperatorOnDevice {
        public:
            using self = DCNV2Forward;
            using supper = OperatorOnDevice;

            DCNV2Forward();

            void init() override;

            /**
             *
             * @param stack Contains x, w
             * @return 1
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void
            forward(const Tensor &x, const Tensor &w, const Tensor &b, const Tensor &offset, const Tensor &mask,
                    const Padding2D &padding, const Stride2D &stride, const Dilation2D &dilation, int deformable_groups,
                    Conv2DFormat format, Tensor &out) = 0;

        private:
            Conv2DFormat m_format;
            std::valarray<int> m_padding4x2;
            int m_deformable_groups;
            std::valarray<int> m_stride4;
            std::valarray<int> m_dilation4;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_DCN_V2_FORWARD_H
