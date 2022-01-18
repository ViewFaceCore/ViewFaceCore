//
// Created by kier on 2019/2/15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_BATCHTOSPACE4D_H
#define TENSORSTACK_BACKEND_BASE_BASE_BATCHTOSPACE4D_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        
       
        class BatchToSpace4D : public OperatorOnDevice {
        public:
            using self = BatchToSpace4D;
            using supper = OperatorOnDevice;

            BatchToSpace4D();

            void CaculateOutputSize(const Shape &input_shape, Shape &output_shape, const int crop_top, const int crop_bottom,
                                    const int crop_left,const int crop_right, const int block_height, const int block_width);
            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void batchtospace4d_run(const Tensor &x,const int crop_top, const int crop_bottom,
                    const int crop_left,const int crop_right, const int block_height, const int block_width, Tensor &out) = 0;

        protected:

            int m_crop[4];
            int m_block_shape[2];
        };
    }
}


#endif
