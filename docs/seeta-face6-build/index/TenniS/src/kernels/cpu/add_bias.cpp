#include <kernels/cpu/add_bias.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

#include "kernels/common/simd.h"
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

/////////////////////////////////////////////////
namespace ts {
    template<typename T>
    void cpu_add_bias_compute_run(const Tensor &x, const Tensor &b, int dim, Tensor &out) {
        const Shape &shape = x.sizes();
        int pre_dims = 1;
        int back_dims = 1;
        for (int i = 0; i < dim; i++) {
            pre_dims *= shape[i];
        }

        for (int i = dim + 1; i < shape.size(); i++) {
            back_dims *= shape[i];
        }

        const T *psrc = x.data<T>();
        const T *pbias = b.data<T>();
        T *pdst = out.data<T>();

        // only used in CPU
        std::memcpy(pdst, psrc, out.count() * sizeof(T));

        int stridedims = back_dims * shape[dim];
        int offset = 0;
        for (int i = 0; i < pre_dims; i++) {
            for (int k = 0; k < shape[dim]; k++) {
                offset = i * stridedims + k * back_dims;
                for (int m = 0; m < back_dims; m++) {
                    pdst[offset + m] += pbias[k];
                }
            }
        }
    }

    template<>
    void cpu_add_bias_compute_run<float>(const Tensor &x, const Tensor &b, int dim, Tensor &out) {
        const Shape &shape = x.sizes();
        int pre_dims = 1;
        int back_dims = 1;
        for (int i = 0; i < dim; i++) {
            pre_dims *= shape[i];
        }

        for (int i = dim + 1; i < shape.size(); i++) {
            back_dims *= shape[i];
        }

        const float *psrc = x.data<float>();
        const float *pbias = b.data<float>();
        float *pdst = out.data<float>();

        // only used in CPU
        //std::memcpy(pdst, psrc, out.count() * sizeof(float));

        const int stridedims = back_dims * shape[dim];
        for (int i = 0; i < pre_dims; i++) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
            for (int k = 0; k < shape[dim]; k++) {
                float32x4 bias_x4(pbias[k]);
                auto offset = i * stridedims + k * back_dims;
                for (int m = 0; m < back_dims - 3; m += 4) {
                    float32x4 src_x4(&psrc[offset + m]);
                    float32x4 dst_x4 = src_x4 + bias_x4;
                    dst_x4.store(&pdst[m + offset]);
                }
                for (int m = back_dims/4*4; m < back_dims; m++) {
                    pdst[offset + m] = psrc[offset + m] + pbias[k];
                }
            }
        }
    }

    void cpu::AddBias::add(const Tensor &x, const Tensor &b, int dim, Tensor &out) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = out.dtype();
        switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_add_bias_compute_run<TYPE>(x, b, dim, out); break; }
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
/////////////////////////////////////////////////

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(AddBias, CPU, name::layer::add_bias())

