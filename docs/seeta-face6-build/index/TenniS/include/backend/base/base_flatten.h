//
// Created by kier on 2019/2/18.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H
#define TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H

#include "base_new_shape.h"

namespace ts {
    namespace base {
        class Flatten : public NewShape {
        public:
            using self = Flatten;
            using supper = NewShape;

            Flatten();

            void init() override;

            Shape newshape(const Tensor &x) final;

        private:
            int m_dim = 1;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_FLATTEN_H
