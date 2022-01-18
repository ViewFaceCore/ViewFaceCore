//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_AFFINE_SAMPLE2D_H
#define TENSORSTACK_BACKEND_BASE_BASE_AFFINE_SAMPLE2D_H

#include "backend/common_structure.h"
#include "operator_on_device.h"

namespace ts {
    namespace base {
        enum class AffineOuterMode : int32_t {
            NEAREST = 0,
            VALUE = 1
        };
        
       
        class Affine_Sample2D : public OperatorOnDevice {
        public:
            using self = Affine_Sample2D;
            using supper = OperatorOnDevice;

            Affine_Sample2D();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void affine_sample_run(const Tensor &x, float rz00, float rz01, float rz02, float rz10, float rz11, float rz12,
                                           float rz20, float rz21, float rz22, Affine_Sample2DType type, int dim,
                                           AffineOuterMode outer_mode, float outer_value,
                                           Tensor &out) = 0;

        protected:

            Affine_Sample2DType m_type;
            int m_dim;
            AffineOuterMode m_outer_mode = AffineOuterMode::NEAREST;
            float m_outer_value = 0;
        };
    }
}


#endif
