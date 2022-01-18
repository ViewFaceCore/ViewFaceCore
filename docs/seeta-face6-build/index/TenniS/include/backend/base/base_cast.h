//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_CAST_H
#define TENSORSTACK_BACKEND_BASE_BASE_CAST_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Cast : public OperatorOnDevice {
        public:
            using self = Cast;
            using supper = OperatorOnDevice;

            Cast(DTYPE dtype);

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void cast(const Tensor &x, DTYPE dtype, Tensor &out);

        protected:
            void set_dtype(DTYPE dtype);

        private:
            DTYPE m_dtype;
        };

        template <DTYPE _dtype>
        class CastTo : public Cast {
        public:
            using self = CastTo;
            using supper = Cast;

            CastTo() : supper(_dtype) {}
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CAST_H
