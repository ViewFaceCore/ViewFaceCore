#include "kernels/cpu/max.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {


        template <typename T>
        static void cpu_max_compute_run(const Tensor &x, int axis,  Tensor &out) {
            auto &x_shape = x.sizes();
            if(axis < 0) {
                axis += int(x_shape.size());
            }

            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1, std::multiplies<int>());
            int input_width = width * x_shape[axis];


            const T* x_data = x.data<T>();
            T* out_data = out.data<T>();

            T* ptr = out_data; 
            T tmp = T(0);

            int cur_input_width = 0;
            for (int i = 0; i < number; ++i) {
                ptr = out_data + i* width;
                cur_input_width = i * input_width; 
                
                for(int k=0; k<width; k++) {
                    for(int m = 0; m < x_shape[axis]; m++) {
                        int cur_step = m * width;
                        if(m == 0) {
                            tmp = *(x_data + k + cur_step + cur_input_width);
                        }else if(*(x_data + k + cur_step + cur_input_width) > tmp) {
                            tmp = *(x_data + k + cur_step + cur_input_width);
                        }
                    }
                    
                    *(ptr + k) = tmp;
                } 
                
            }
        }


        void Max::max(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_max_compute_run<TYPE>(x, m_dim, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }


        }

    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Max, CPU, name::layer::max())
