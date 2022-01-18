#include <kernels/gpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <global/fp16_operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"
#include "kernels/gpu/math_cublas.h"
#include "kernels/gpu/math_gpu.h"


namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void gpu_inner_prod_compute_run_kernel(int m, int n, int k, const T *A, const T *B, T *C) {
            __shared__ T ds_A[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];
            __shared__ T ds_B[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int Row = by * blockDim.y + ty;
            int Col = bx * blockDim.x + tx;

            T comp = T(0.f);
            T Cvalue = T(0.f);

            for (int t=0; t<(n - 1) / TRANS_BLOCK_DIM + 1; ++t) {
                if (Row < m && t * blockDim.x + tx < n)
                    ds_A[ty][tx] = A[Row*n+t*blockDim.x+tx];
                else
                    ds_A[ty][tx] = T(0.f);

                if (t * blockDim.y + ty < n && Col < k)
                    ds_B[ty][tx] = B[(t*blockDim.y + ty)*k+Col];
                else
                    ds_B[ty][tx] = T(0.f);

                __syncthreads();

                for (int i = 0; i < blockDim.x; ++i) {
                    //Cvalue += ds_A[ty][i] * ds_B[i][tx];
                    T t;
                    comp -= ds_A[ty][i] * ds_B[i][tx];
                    t = Cvalue - comp;
                    comp = (t - Cvalue) + comp;
                    Cvalue = t;
                }

                __syncthreads();

                if(Row < m && Col < k) {
                    C[Row*k+Col]=Cvalue;
                }
            }//end for
        
        
        }


        template<typename T>
        static void gpu_inner_prod_compute_run(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) {
            const Shape &lhs_shape = lhs.sizes();
            const Shape &rhs_shape = rhs.sizes();

            const T *psrc = lhs.data<T>();
            const T *pdot = rhs.data<T>();
            T *pdst = out.data<T>();

#ifdef TS_USE_CUBLAS
            auto &context = ctx::ref<DeviceContext>();
            CUDAContextHandle* handle = reinterpret_cast<CUDAContextHandle*>(context.handle);
            auto cublas_handle = handle->cublas_handle();

            auto rhs_tranpose = transpose ? cublas::Trans : cublas::NoTrans;
            auto N = transpose ? rhs_shape[0] : rhs_shape[1];

            cublas::math<T>::gemm(cublas_handle, cublas::NoTrans, rhs_tranpose,
                lhs_shape[0], N, lhs_shape[1], T(1.f), psrc, pdot, T(0.f), pdst);
            /*cublas::math<T>::gemm(cublas_handle,cublas::RowMajor,cublas::NoTrans, cublas::NoTrans, 
                lhs_shape[0], rhs_shape[1], lhs_shape[1], 1,psrc, lhs_shape[1], pdot, rhs_shape[1], 0,pdst, rhs_shape[1]);*/
            
#else
            auto rhs_tranpose = transpose ? cublas::Trans : cublas::NoTrans;
            auto N = transpose ? rhs_shape[0] : rhs_shape[1];
            gpu::math<T>::gemm(
                    cublas::NoTrans, rhs_tranpose,
                    lhs_shape[0], N, lhs_shape[1], T(1.f), psrc, pdot, T(0.f), pdst);
            /*
            dim3 blocksize(CUDA_BLOCK(rhs_shape[1], TRANS_BLOCK_DIM), CUDA_BLOCK(lhs_shape[0], TRANS_BLOCK_DIM),1);
            dim3 threadsize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM,1);
            RUN_KERNEL(gpu_inner_prod_compute_run_kernel<T>, blocksize, threadsize, lhs_shape[0], lhs_shape[1], rhs_shape[1], psrc, pdot, pdst);
             */
#endif
        }

        void InnerProd::inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_inner_prod_compute_run<TYPE>(lhs, rhs, transpose, out); break; }
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

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(InnerProd, GPU, name::layer::inner_prod())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(InnerProd, GPU, name::layer::inner_prod())
#endif
