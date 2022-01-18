#include <backend/base/base_non_max_suppression_v3.h>

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Non_Max_Suppression_V3::Non_Max_Suppression_V3() {
            field(name::max_output_size,REQUIRED);
            field(name::iou_threshold,REQUIRED);
            field(name::score_threshold,REQUIRED);
            field(name::mode,REQUIRED);
        }

        void Non_Max_Suppression_V3::init() {
            supper::init();

            Tensor mode_tensor =  get(name::mode);
            m_mode = tensor::to_string(mode_tensor);
            
            Tensor max_output_size_tensor =  get(name::max_output_size);
            m_max_output_size = tensor::to_int(max_output_size_tensor);
            Tensor iou_tensor =  get(name::iou_threshold);
            m_iou_threshold = tensor::to_float(iou_tensor);
            Tensor score_tensor =  get(name::score_threshold);
            m_score_threshold = tensor::to_float(score_tensor);

            TS_AUTO_CHECK(m_max_output_size > 0); 
        }


        int Non_Max_Suppression_V3::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 2);
             
            auto &x = stack[0];
            Shape x_shape = x.sizes();
            TS_AUTO_CHECK(x_shape.size() == 2);
            TS_AUTO_CHECK(x_shape[1] == 4);

            auto &score = stack[1];
            Shape score_shape = score.sizes();
            TS_AUTO_CHECK(score_shape.size() == 1);
            TS_AUTO_CHECK(score_shape[0] == x_shape[0]);

            TS_AUTO_CHECK(x.dtype() == FLOAT32);

            Shape shape;
            shape.resize(1);
            shape[0] = m_max_output_size;
            output.resize(1);
            output[0] = Tensor::Prototype(INT32, shape);

            return 1;
        }

        int Non_Max_Suppression_V3::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 2);

            auto memory_device = running_memory_device();
            auto box = stack[0].view(memory_device);
            auto scores = stack[1].view(memory_device);

            Shape shape;
            shape.resize(1);
            shape[0] = m_max_output_size;

            auto output_proto = Tensor::Prototype(INT32, shape);
            auto &out = *stack.push(output_proto, memory_device);
            non_max_suppression_v3(box, scores, out);

            return 1;
        }
    }

}
