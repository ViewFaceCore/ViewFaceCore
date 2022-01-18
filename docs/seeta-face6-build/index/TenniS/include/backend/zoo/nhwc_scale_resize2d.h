//
// Created by kier on 2019/3/18.
//

#ifndef TENSORSTACK_NHWC_SCALE_RESIZE2D_H
#define TENSORSTACK_NHWC_SCALE_RESIZE2D_H

#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class NHWCScaleResize2D : public Operator {
        public:
            using self = NHWCScaleResize2D;
            using supper = Operator;

            NHWCScaleResize2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::vector<int> m_size;    // {width, height} format
            Operator::shared m_resize2d_op;

            Tensor m_dynamic_size;
        };
    }
}


#endif //TENSORSTACK_NHWC_SCALE_RESIZE2D_H
