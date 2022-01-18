//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_GLOBAL_POOLING2D_H
#define TENSORSTACK_BACKEND_BASE_BASE_GLOBAL_POOLING2D_H

#include "operator_on_device.h"
#include <valarray>

#include "backend/common_structure.h"
#include "base_pooling2d_core.h"

namespace ts {
    namespace base {
        class GlobalPooling2D : public OperatorOnDevice, public Pooling2DCore {
        public:
            using self = GlobalPooling2D;
            using supper = OperatorOnDevice;

            GlobalPooling2D();

            void init() override;

            /**
             *
             * @param stack Contains x
             * @return 1
             */
            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            Conv2DFormat m_format;
            Pooling2DType m_type;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_GLOBAL_POOLING2D_H
