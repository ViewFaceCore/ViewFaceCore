#include <kernels/cpu/quantize.h>
#include <algorithm>
#include <math.h>

#include "backend/name.h"
#include "global/operator_factory.h"

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include "kernels/common/openmp.h"
#endif

namespace ts {
    namespace cpu {
        template<typename T>
        static inline int8_t to_int8(const T src) {
            if (src > 127) return 127;
            if (src < -128) return -128;
            return static_cast<int8_t>(src);
        }

        template<>
        inline int8_t to_int8<float>(const float src) {
            auto int32_temp = int32_t(round(src));
            if (int32_temp > 127) return 127;
            if (int32_temp < -128) return -128;
            return static_cast<int8_t>(int32_temp);
        }

        template<>
        inline int8_t to_int8<double>(const double src) {
            auto int32_temp = int32_t(round(src));
            if (int32_temp > 127) return 127;
            if (int32_temp < -128) return -128;
            return static_cast<int8_t>(int32_temp);
        }

        template<typename T>
        static void cpu_quantize_compute_run(const Tensor &x, std::vector<float> quantize_scales, Tensor &out) {
            const T *input_data = x.data<T>();
            int8_t *output_data = out.data<dtype<INT8>::declare>();
            int count = out.count();
            auto quantize_group = quantize_scales.size();
            float quantize_scale;
            if (quantize_group == 1) {
                quantize_scale = quantize_scales[0];
#ifdef TS_USE_OPENMP
                #pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int i = 0; i < count; i++) {
                    output_data[i] = to_int8(input_data[i] * quantize_scale);
                }
            }
            else {
                auto loop_count = int(std::ceil(static_cast<float>(count) / quantize_group));
                int index = 0;
                for (size_t n = 0; n < quantize_group; n++){
                    quantize_scale = quantize_scales[n];
                    int loop_count_temp = loop_count;
                    while (index < count && loop_count_temp) {
                        output_data[index] = to_int8(input_data[index] * quantize_scale);
                        index++;
                        loop_count_temp--;
                    }
                }
            }
        }

        template<>
        void cpu_quantize_compute_run<float>(const Tensor &x, std::vector<float> quantize_scales, Tensor &out) {
            const float *input_data = x.data<float>();
            int8_t *output_data = out.data<dtype<INT8>::declare>();
            int count = out.count();
            auto quantize_group = quantize_scales.size();
            float quantize_scale;
            if (quantize_group == 1) {
                quantize_scale = quantize_scales[0];
            int count_4 = count >> 3;
            int remain = count_4 << 3;

            float32x4x2 scale_x4x2(quantize_scale);
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int i = 0; i < count_4; i++){
                    int ii = i * 8;
                    float32x4x2 input_x4x2(&input_data[ii]);  
                    float32x4x2 output_x4x2 = input_x4x2 * scale_x4x2;
                    int32x4x2 output_int_x4x2 = floatx4x2_to_int32x4x2(output_x4x2);
                    auto *output_int32x8 = (int32_t*)(void*)(&output_int_x4x2.value);
                    *(output_data + ii) = to_int8(output_int32x8[0]);
                    *(output_data + ii + 1) = to_int8(output_int32x8[1]);
                    *(output_data + ii + 2) = to_int8(output_int32x8[2]);
                    *(output_data + ii + 3) = to_int8(output_int32x8[3]);
                    *(output_data + ii + 4) = to_int8(output_int32x8[4]);
                    *(output_data + ii + 5) = to_int8(output_int32x8[5]);
                    *(output_data + ii + 6) = to_int8(output_int32x8[6]);
                    *(output_data + ii + 7) = to_int8(output_int32x8[7]);
                    //*(output_data + ii) = to_int8(*((float*)&(output_x4x2.value)));
                    //*(output_data + ii + 1) = to_int8(*(((float*)&(output_x4x2.value)) + 1));
                    //*(output_data + ii + 2) = to_int8(*(((float*)&(output_x4x2.value)) + 2));
                    //*(output_data + ii + 3) = to_int8(*(((float*)&(output_x4x2.value)) + 3));
                    //*(output_data + ii + 4) = to_int8(*(((float*)&(output_x4x2.value)) + 4));
                    //*(output_data + ii + 5) = to_int8(*(((float*)&(output_x4x2.value)) + 5));
                    //*(output_data + ii + 6) = to_int8(*(((float*)&(output_x4x2.value)) + 6));
                    //*(output_data + ii + 7) = to_int8(*(((float*)&(output_x4x2.value)) + 7));
                }
                for (int i = remain; i < count; i++){
                    output_data[i] = to_int8(input_data[i] * quantize_scale);
                }
            }
            else {
                auto loop_count = int(std::ceil(static_cast<float>(count) / quantize_group));
                int index = 0;
                for (size_t n = 0; n < quantize_group; n++) {
                    quantize_scale = quantize_scales[n];
                    int loop_count_temp = loop_count;
                    while (index < count && loop_count_temp) {
                        output_data[index] = to_int8(input_data[index] * quantize_scale);
                        index++;
                        loop_count_temp--;
                    }
                }
            }
        }

        void Quantize::quantize(const Tensor &x, std::vector<float> quantize_scales, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = x.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_quantize_compute_run<TYPE>(x, quantize_scales, out); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << this->op() << " not support this data type: " << dtype << eject;
                break;
            }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Quantize, ts::CPU, name::layer::quantize())
