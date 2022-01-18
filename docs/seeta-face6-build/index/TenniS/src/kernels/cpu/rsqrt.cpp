#include <kernels/cpu/rsqrt.h>
#include <algorithm>

#include "backend/name.h"
#include "global/operator_factory.h"


namespace ts {
    namespace cpu {


/*
double InvSqrt(double number)
{
      double x2, y;
      const double threehalfs = 1.5F;
      union
      {
            double d;
            __int64 i;
      }d;
      x2 = number * 0.5F;
      y = number;
      d.d =  y;
      d.i = 0x5fe6ec85e7de30da - (d.i >> 1);
      y = d.d;
      y = y * (threehalfs - (x2 * y * y)); //1stiteration
      y = y * (threehalfs - (x2 * y * y)); //2nditeration, this can be removed
      return y;
}
*/
        template<typename T>
        static void cpu_rsqrt_compute_run(const Tensor &x, Tensor &out) {
            const T *input_data = x.data<T>();
            T *output_data = out.data<T>();
            int count = out.count();

            std::memcpy(output_data, input_data, count * sizeof(T));

            for (int i = 0; i < count; i++) {
                T val = *output_data;
                *output_data = T(1. / (sqrt(val)));
                output_data++;
            }
        }


        void Rsqrt::active(const Tensor &x, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_rsqrt_compute_run<TYPE>(x, out); break; }
/*
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
*/
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
TS_REGISTER_OPERATOR(Rsqrt, ts::CPU, name::layer::rsqrt())
