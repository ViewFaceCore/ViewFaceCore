//
// Created by kier on 2019/2/20.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_CAST_V2_H
#define TENSORSTACK_BACKEND_BASE_BASE_CAST_V2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class CastV2 : public OperatorOnDevice {
        public:
            using self = CastV2;
            using supper = OperatorOnDevice;

            CastV2();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void cast(const Tensor &x, DTYPE dtype, Tensor &out);

            void set_dtype(DTYPE dtype);
            DTYPE get_dtype() const;

        private:
            DTYPE m_dtype;
        };

    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CAST_V2_H
