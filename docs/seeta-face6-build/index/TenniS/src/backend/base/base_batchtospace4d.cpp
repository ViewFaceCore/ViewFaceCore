//
// Created by kier on 2019/2/15.
//

#include "backend/base/base_batchtospace4d.h"

#include <utils/assert.h>
#include <numeric>

#include <backend/name.h>
#include <core/tensor_builder.h>


namespace ts {
    namespace base {


        void BatchToSpace4D::CaculateOutputSize(const Shape &input_shape, Shape &output_shape, const int block_height, const int block_width,
                                                const int crop_top, const int crop_bottom,
                                                const int crop_left,const int crop_right) {
            output_shape[0] = input_shape[0] / (block_height * block_width);
            output_shape[2] = input_shape[2] * block_height - crop_top - crop_bottom;
            output_shape[3] = input_shape[3] * block_width - crop_left - crop_right;
            output_shape[1] = input_shape[1];
        }


        BatchToSpace4D::BatchToSpace4D() {
            field(name::crop, REQUIRED);
            field(name::block_shape, REQUIRED);
        }

        void BatchToSpace4D::init() {
            supper::init();

            Tensor crop_tensor = tensor::cast(INT32, get(name::crop));
            Tensor block_shape_tensor = tensor::cast(INT32, get(name::block_shape));

            TS_AUTO_CHECK(crop_tensor.has_shape({2,2}));
            TS_AUTO_CHECK(block_shape_tensor.has_shape({2,}));

            for(size_t i=0; i<4; i++) {
                m_crop[i] = crop_tensor.data<int32_t>(i);
            }

            m_block_shape[0] = block_shape_tensor.data<int32_t>(0);
            m_block_shape[1] = block_shape_tensor.data<int32_t>(1);
            TS_AUTO_CHECK((m_block_shape[0] >= 1) && (m_block_shape[1] >= 1));
        }

        int BatchToSpace4D::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);
            auto &x = stack[0];

            Shape input_shape = x.sizes();
            TS_AUTO_CHECK(input_shape.size() == 4);
            TS_AUTO_CHECK((input_shape[0] > 0) && (input_shape[0] % (m_block_shape[0] * m_block_shape[1]) == 0));

            Shape output_shape;
            output_shape.resize(4);
            CaculateOutputSize(input_shape, output_shape,m_block_shape[0],m_block_shape[1],m_crop[0],m_crop[1],m_crop[2],m_crop[3]);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), output_shape);

            return 1;
        }

        int BatchToSpace4D::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 1);

            std::vector<Tensor::Prototype> output;
            infer(stack, output);
            

            auto memory_device = running_memory_device();
            auto x = stack[0].view(memory_device);

            auto out = *stack.push(output[0], memory_device);
            batchtospace4d_run(x, m_crop[0],m_crop[1],m_crop[2],m_crop[3],m_block_shape[0],m_block_shape[1], out);

            return 1;
        }
    }
}
