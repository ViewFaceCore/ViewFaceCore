//
// Created by kier on 2019-05-28.
//

#ifndef TENSORSTACK_BACKEND_ZOO_LETTERBOX_H
#define TENSORSTACK_BACKEND_ZOO_LETTERBOX_H


#include <runtime/operator.h>
#include "backend/common_structure.h"

namespace ts {
    namespace zoo {
        class NHWCLetterBox : public Operator {
        public:
            using self = NHWCLetterBox;
            using supper = Operator;

            NHWCLetterBox();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

        private:
            std::vector<int> m_size;    ///< {width, height} format
            Operator::shared m_sample_op;

            Tensor m_sample_size;
            Tensor m_sample_affine;
        };
    }
}

#endif //TENSORSTACK_BACKEND_ZOO_LETTERBOX_H
