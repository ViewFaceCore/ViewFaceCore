//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_RESIZE2D_H
#define TENSORSTACK_BACKEND_BASE_BASE_RESIZE2D_H

#include "operator_on_device.h"
#include "backend/common_structure.h"

namespace ts {
    namespace base {
        class Resize2D : public OperatorOnDevice {
        public:
            using self = Resize2D;
            using supper = OperatorOnDevice;

            Resize2D();

            void init() override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param stack contains: x, size
             * @return
             */
            int run(Stack &stack) override;

            /**
             *
             * @param x
             * @param dim the dim and dim + 1 is resized new height and width
             * @param type
             * @param out
             */
            virtual void resize2d(const Tensor &x, int dim, Resize2DType type, Tensor &out) = 0;

        private:
            Resize2DType m_type = Resize2DType::LINEAR;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_RESIZE2D_H
