//
// Created by kier on 2019/3/5.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_UNSQUEEZE_H
#define TENSORSTACK_BACKEND_BASE_BASE_UNSQUEEZE_H

#include "base_new_shape.h"

namespace ts {
    namespace base {
        class Unsqueeze : public NewShape {
        public:
            using self = Unsqueeze;
            using supper = NewShape;

            Unsqueeze();

            void init() override;

            Shape newshape(const Tensor &x) override;

        private:
            std::vector<int> m_axes;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_UNSQUEEZE_H
