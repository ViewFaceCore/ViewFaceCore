#include "kernels/cpu/slice.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {


        template <typename T>
        static void cpu_slice_compute_run(const Tensor &x, const std::vector<int> &begins,const std::vector<int> &sizes, Tensor &out) {
            auto &x_shape = x.sizes();

            T * p_outdata = out.data<T>();
            const T* p_xdata  = x.data<T>();

            Shape out_shape = out.sizes();
            HypeShape hype_x_shape(x_shape);
            HypeShape hype_out_shape(out_shape);

         
            for(int i=0; i<out.count(); i++) {
                Shape tmp_shape = hype_out_shape.to_coordinate(i); 
                for(size_t k = 0; k < tmp_shape.size(); k++) {
                    tmp_shape[k] += begins[k];
                }       
                int index = hype_x_shape.to_index(tmp_shape);
                p_outdata[i] = p_xdata[index];
            }
        }


        void Slice::slice(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_slice_compute_run<TYPE>(x, m_begin, m_size, out); break; }
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
TS_REGISTER_OPERATOR(Slice, CPU, name::layer::slice())
