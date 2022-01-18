#include <kernels/gpu/add_bias.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include "global/fp16_operator_factory.h"
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/gpu/gpu_kernel.h"

/////////////////////////////////////////////////
namespace ts {
    namespace gpu {
    template<typename T>
    static __global__ void add_bias_kernel(const T* base, T* data, int size, int step, int slice,
                                        const T* bias, int biaslen ) {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < size) {
            int dim = index % ( step * slice ) / (step);
            data[index] = base[index] + bias[dim];
        }
    }

    template<typename T>
    static void gpu_add_bias_compute_run(const Tensor &x, const Tensor &b, int dim, Tensor &out) {
        const Shape &shape = x.sizes();
        //int pre_dims = 1;
        int back_dims = 1;
        //for (int i = 0; i < dim; i++) {
        //    pre_dims *= shape[i];
        //}

        for (int i = dim + 1; i < shape.size(); i++) {
            back_dims *= shape[i];
        }

        const T *psrc = x.data<T>();
        const T *pbias = b.data<T>();
        T *pdst = out.data<T>();

//        memcpy((void*)pdst, out.device(), x.count() * sizeof(T),
//               (void*)psrc, x.device(), x.count() * sizeof(T));


        RUN_KERNEL(add_bias_kernel<T>, CUDA_BLOCK(x.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                   psrc, pdst, x.count(), back_dims, shape[dim], pbias, b.count());

        //cudaDeviceSynchronize();
    }

    void AddBias::add(const Tensor &x, const Tensor &b, int dim, Tensor &out) {
        // Notice: the all tensor' memory device are CPU, as given in running_memory_device
        DTYPE dtype = out.dtype();
        switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_add_bias_compute_run<TYPE>(x, b, dim, out); break; }
            DECLARE_COMPUTE_RUN(INT8, int8_t);
            DECLARE_COMPUTE_RUN(UINT8, uint8_t);
            DECLARE_COMPUTE_RUN(INT16, int16_t);
            DECLARE_COMPUTE_RUN(UINT16, uint16_t);
            DECLARE_COMPUTE_RUN(INT32, int32_t);
            DECLARE_COMPUTE_RUN(UINT32, uint32_t);
            DECLARE_COMPUTE_RUN(INT64, int64_t);
            DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
            DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
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
/////////////////////////////////////////////////

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(AddBias, GPU, name::layer::add_bias())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(AddBias, ts::GPU, name::layer::add_bias())
#endif

