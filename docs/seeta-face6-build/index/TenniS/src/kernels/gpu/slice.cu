#include "kernels/gpu/slice.h"
#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
#include "backend/name.h"

#include "kernels/gpu/gpu_kernel.h"

#include <numeric>
#include <cuda_fp16.h>

namespace ts {
    namespace gpu {

        template <typename T>
        static __global__ void gpu_slice_kernel(const T* x_data, const int32_t* b_data,
                                T* out_data, int count, int dims, GpuHypeShape x_shape, GpuHypeShape out_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            auto x_weight_it = x_shape.weights + 1;
            auto out_weight_it = out_shape.weights + 1;

            int in_index = index;
            int cur_index = 0;


            for(int i=0; i<dims - 1; i++) {
                cur_index += ((in_index / (*out_weight_it)) + b_data[i]) * (*x_weight_it);
                in_index %= *out_weight_it;
                ++x_weight_it;
                ++out_weight_it;
            }

            cur_index += in_index + b_data[dims -1];
            out_data[index] = x_data[cur_index];

        }


        template <typename T>
        static void gpu_slice_compute_run(const Tensor &x, const std::vector<int> &begins,const std::vector<int> &sizes, Tensor &out) {
            auto &x_shape = x.sizes();

            T * p_outdata = out.data<T>();
            const T* p_xdata  = x.data<T>();

            Shape out_shape = out.sizes();

            auto gpu_hype_xshape = MakeGPUHypeShape(x.device(), {x_shape});
            auto gpu_hype_outshape = MakeGPUHypeShape(x.device(), {out.sizes()});

            auto & x_hype_shape = gpu_hype_xshape.second[0];
            auto & out_hype_shape = gpu_hype_outshape.second[0];


            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = int(begins.size());
            Tensor tmp_tensor(out.device(), INT32, tmpshape);
            int * p_bdata = tmp_tensor.data<int32_t>();

            memcpy((void*)p_bdata, out.device(), begins.size() * sizeof(int32_t),
                   (void*)begins.data(), MemoryDevice(CPU), begins.size() * sizeof(int32_t));


            int number = out.count();
            dim3 blockSize(CUDA_THREAD_NUM);
            dim3 gridSize(CUDA_BLOCK(number, blockSize.x));

            // int dims = int(x_shape.size());
            RUN_KERNEL(gpu_slice_kernel<T>, gridSize, blockSize,
                       p_xdata, p_bdata, p_outdata, number, int(x_shape.size()), x_hype_shape, out_hype_shape);

        }


        void Slice::slice(const Tensor &x, Tensor &out) {
            DTYPE dtype = out.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_slice_compute_run<TYPE>(x, m_begin, m_size, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT16, half);
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
using namespace gpu;
TS_REGISTER_OPERATOR(Slice, GPU, name::layer::slice())
TS_REGISTER_FP16_OPERATOR(Slice, GPU, name::layer::slice())
