#include <kernels/cpu/fused_batch_norm.h>
#include <core/tensor_builder.h>

#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>
#include <vector>

#ifdef TS_USE_SIMD
#include "kernels/common/simd.h"
#endif

namespace ts {
    namespace cpu {

        template<typename T>
        static void cpu_batch_norm_compute_run(const Tensor &x,
                                               const Tensor &mean, const Tensor &variance,
                                               const Tensor &scale, const Tensor &bias,
                                               int dim, float epsilon, Tensor &out) {
            const Shape &shape = x.sizes();
            int predims = 1;
            int backdims = 1;
            for (int i = 0; i < dim; i++) {
                predims *= shape[i];
            }

            for (int i = dim + 1; i < shape.size(); i++) {
                backdims *= shape[i];
            }

            const T *psrc = x.data<T>();
            const T *pmean = mean.data<T>();
            const T *pvariance = variance.data<T>();
            const T *pscale = scale.data<T>();
            const T *pbias = bias.data<T>();
            T *pdst = out.data<T>();

            int stridedims = backdims * shape[dim];
            int offset = 0;

            std::vector<T> vec(variance.count());
            for (size_t i = 0; i < vec.size(); i++) {
                vec[i] = T(T(1) / sqrt(pvariance[i] + T(epsilon)));
            }

            for (int i = 0; i < predims; i++) {
                for (int k = 0; k < shape[dim]; k++) {
                    offset = i * stridedims + k * backdims;
                    T mean_val = pmean[k];
                    T vec_val = vec[k];
                    T scale_val = pscale[k];
                    T bias_val = pbias[k];
                    T *pdst_temp = pdst + offset;
                    const T *psrc_temp = psrc + offset;
                    for (int m = 0; m < backdims; m++) {
                        *pdst_temp = (*psrc_temp - mean_val) * vec_val * scale_val + bias_val;
                        pdst_temp++;
                        psrc_temp++;
                    }
                }
            }
        }

#ifdef TS_USE_SIMD
        template<>
        void cpu_batch_norm_compute_run<float>(const Tensor &x,
            const Tensor &mean, const Tensor &variance,
            const Tensor &scale, const Tensor &bias,
            int dim, float epsilon, Tensor &out) {
            const Shape &shape = x.sizes();
            int predims = 1;
            int backdims = 1;
            for (int i = 0; i < dim; i++) {
                predims *= shape[i];
            }

            for (int i = dim + 1; i < shape.size(); i++) {
                backdims *= shape[i];
            }

            const float *psrc = x.data<float>();
            const float *pmean = mean.data<float>();
            const float *pvariance = variance.data<float>();
            const float *pscale = scale.data<float>();
            const float *pbias = bias.data<float>();
            float *pdst = out.data<float>();

            // only used in CPU
            //std::memcpy(pdst, psrc, out.count() * sizeof(float));

            int stridedims = backdims * shape[dim];
            int offset = 0;

            std::vector<float> vec(variance.count());
            for (size_t i = 0; i < vec.size(); i++) {
                vec[i] = float(1) / sqrt(pvariance[i] + float(epsilon));
            }

            for (int i = 0; i < predims; i++) {
                for (int k = 0; k < shape[dim]; k++) {
                    offset = i * stridedims + k * backdims;
                    float mean_val = pmean[k];
                    float vec_val = vec[k];
                    float scale_val = pscale[k];
                    float bias_val = pbias[k];
                    float32x4 mean_val_x4(mean_val);
                    float32x4 vec_val_x4(vec_val);
                    float32x4 scale_val_x4(scale_val);
                    float32x4 bias_val_x4(bias_val);
                    for (int m = 0; m < backdims - 3; m += 4) {
                        float32x4 psrc_x4(&psrc[m + offset]);
                        float32x4 pdst_x4 = (psrc_x4 - mean_val_x4) * vec_val_x4 * scale_val_x4 + bias_val_x4;
                        pdst_x4.store(&pdst[m + offset]);
                    }
                    for (int m = backdims/4*4; m < backdims; m++) {
                        pdst[m + offset] = (psrc[m + offset] - mean_val) * vec_val * scale_val + bias_val;
                    }
                }
            }
        }
#endif

        void FusedBatchNorm::batch_norm(const Tensor &x, const Tensor &mean, const Tensor &variance,
                                   const Tensor &scale, const Tensor &bias,
                                   int dim, float epsilon, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_batch_norm_compute_run<TYPE>(x, mean, variance, scale, bias, dim, epsilon, out); break; }
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
TS_REGISTER_OPERATOR(FusedBatchNorm, CPU, name::layer::fused_batch_norm())
