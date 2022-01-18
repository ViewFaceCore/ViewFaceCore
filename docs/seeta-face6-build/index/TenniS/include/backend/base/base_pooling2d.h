//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_POOLING2D_H
#define TENSORSTACK_BACKEND_BASE_BASE_POOLING2D_H

#include "operator_on_device.h"
#include <valarray>

#include "backend/common_structure.h"
#include "base_pooling2d_core.h"

namespace ts {
    namespace base {
        class Pooling2D : public OperatorOnDevice, public Pooling2DCore {
        public:
            using self = Pooling2D;
            using supper = OperatorOnDevice;

            Pooling2D();

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
            std::valarray<int> m_padding4x2;
            Padding2DType m_padding_type;
            std::valarray<int> m_ksize4;
            std::valarray<int> m_stride4;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_POOLING2D_H
