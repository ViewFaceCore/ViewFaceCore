//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_H
#define TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_H

#include "base_new_shape.h"

namespace ts {
    namespace base {
        class Reshape : public NewShape {
        public:
            using self = Reshape;
            using supper = NewShape;

            Reshape();

            void init() override;

            Shape newshape(const Tensor &x) override;

        private:
            Shape m_shape;
            int m_broadcast_dim = -1;
            int m_count_without_dim = -1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_RESHAPE_H
