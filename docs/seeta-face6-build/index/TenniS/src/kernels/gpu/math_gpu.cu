#include "kernels/common/math.h"
#include "kernels/gpu/math_gpu.h"
#include "utils/assert.h"
#include "utils/ctxmgr.h"

#include <iostream>
#include <cmath>

#include "kernels/gpu/memory_gpu.h"
#include "kernels/gpu/operator_on_gpu.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"

#include "kernels/gpu/cudax_fp16_math.h"
#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace gpu {

        template<typename T>
        __global__ static void gemm_kernel(
            bool transA, bool transB,
            int M, int N, int K,
            T alpha,
            const T *A, int lda,
            const T *B, int ldb,
            T beta,
            T *C, int ldc) {

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

            for (int t = 0; t< (K - 1) / TRANS_BLOCK_DIM + 1; ++t) {
                if (Row < M && t * blockDim.x + tx < K) {
                    if (transA)
                        ds_A[tx][ty] = alpha * A[tx*lda + t*blockDim.x + Row];
                    else
                        ds_A[tx][ty] = alpha * A[Row*lda + t*blockDim.x + tx];
                }
                else
                    ds_A[tx][ty] = T(0.f);

                if (t * blockDim.y + ty < K && Col < N)
                    if (transB)
                        ds_B[tx][ty] = B[(t*blockDim.y + Col)*ldb + ty];
                    else
                        ds_B[tx][ty] = B[(t*blockDim.y + ty)*ldb + Col];
                else
                    ds_B[tx][ty] = T(0.f);

                __syncthreads();

                for (int i = 0; i < blockDim.x; ++i) {
                    //Cvalue += ds_A[i][ty] * ds_B[tx][i];
                    T t;
                    comp -= ds_A[i][ty] * ds_B[tx][i];
                    t = Cvalue - comp;
                    comp = (t - Cvalue) + comp;
                    Cvalue = t;
                }

                __syncthreads();

                if (Row < M && Col < N) {
                    C[Row*N + Col] = beta * C[Row*N + Col] + Cvalue;
                }
            }//end for

        }

        template<typename T>
        __global__ static void abs_sum_kernel(const int N, T *x, T * z) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            __shared__ T cache[CUDA_THREAD_NUM];
            //extern __shared__ T cache[];

            int cache_index = threadIdx.x;
            T temp = T(0);
            for (; index < N ; index += blockDim.x * gridDim.x) {
                temp += fabs(x[index]);
            }
            cache[cache_index] = temp;

            __syncthreads();

            unsigned int floor_pow = blockDim.x;
            if (floor_pow & (floor_pow - 1))
            {
                while (floor_pow & (floor_pow - 1))
                {
                    floor_pow &= (floor_pow - 1);
                }
                if (cache_index >= floor_pow)
                {
                    cache[cache_index - floor_pow] += cache[cache_index];
                }
                __syncthreads();
            }

            for (int i = floor_pow / 2; i > 0; i /= 2)
            {
                if (cache_index < i)
                {
                    cache[cache_index] += cache[cache_index + i];
                }
                __syncthreads();
            }
            if (cache_index == 0) { 
                z[blockIdx.x] = cache[0];
            }
            //if (cache_index == 0) {
            //    volatile T* temp = cache;
            //    z[blockIdx.x] = temp[0];
            //}
        }

        template<typename T>
        __global__ static void sum_kernel(const int N, T *x, T * z) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            __shared__ T cache[CUDA_THREAD_NUM];

            int cache_index = threadIdx.x;
            T temp = T(0.f);
            for (; index < N ; index += blockDim.x * gridDim.x) {
                temp += x[index];
            }
            cache[cache_index] = temp;

            __syncthreads();

            unsigned int floor_pow = blockDim.x;
            if (floor_pow & (floor_pow - 1))
            {
                while (floor_pow & (floor_pow - 1))
                {
                    floor_pow &= (floor_pow - 1);
                }
                if (cache_index >= floor_pow)
                {
                    cache[cache_index - floor_pow] += cache[cache_index];
                }
                __syncthreads();
            }

            for (int i = floor_pow / 2; i > 0; i /= 2)
            {
                if (cache_index < i)
                {
                    cache[cache_index] += cache[cache_index + i];
                }
                __syncthreads();
            }

            if (cache_index == 0) {
                z[blockIdx.x] = cache[0];
            }    
        }

        template<typename T>
        __global__ static void dot_kernel(const int N, const T *x, const T *y,T * z) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            __shared__ T cache[CUDA_THREAD_NUM];

            int cache_index = threadIdx.x;
            T temp = T(0.f);
            for (; index < N; index += blockDim.x * gridDim.x) {
                temp += (x[index] * y[index]);
            }
            cache[cache_index] = temp;

            __syncthreads();

            unsigned int floor_pow = blockDim.x;
            if (floor_pow & (floor_pow - 1))
            {
                while (floor_pow & (floor_pow - 1))
                {
                    floor_pow &= (floor_pow - 1);
                }
                if (cache_index >= floor_pow)
                {
                    cache[cache_index - floor_pow] += cache[cache_index];
                }
                __syncthreads();
            }

            for (int i = floor_pow / 2; i > 0; i /= 2)
            {
                if (cache_index < i)
                {
                    cache[cache_index] += cache[cache_index + i];
                }
                __syncthreads();
            }

            if (cache_index == 0)
                z[blockIdx.x] = cache[0];
        }

        template<typename T>
        bool math<T>::dot(const int N, const T *x, const T *y, T *z) {
            int grid_size = CUDA_BLOCK(N, CUDA_THREAD_NUM);

            Tensor tmp_tensor(Tensor::InFlow::DEVICE, UINT8, {int32_t(grid_size *sizeof(T))});
            T* tmp_z = tmp_tensor.data<T>();
            RUN_KERNEL(dot_kernel<T>, CUDA_BLOCK(N, CUDA_THREAD_NUM), CUDA_THREAD_NUM, N,x,y,tmp_z);
            while (grid_size > CUDA_THREAD_NUM) {
                int len = grid_size;
                grid_size = CUDA_BLOCK(grid_size, CUDA_THREAD_NUM);
                RUN_KERNEL(sum_kernel<T>, grid_size, CUDA_THREAD_NUM, len, tmp_z, tmp_z);
            }

            RUN_KERNEL(sum_kernel<T>, 1, grid_size, grid_size,tmp_z,z);
            return true;
        }

        //template<typename T>
        //bool math<T>::dot(int N, const T *x, const T *y, T *z) {
        //    return dot(N, x, 1, y, 1, z);
        //}

        template<typename T>
        inline bool inline_gemm_row_major(
            cublas::Transpose TransA,
            cublas::Transpose TransB,
            int M, int N, int K,
            T alpha,
            const T *A, int lda,
            const T *B, int ldb,
            T beta,
            T *C, int ldc) {
            TS_AUTO_CHECK(lda == (TransA == cublas::NoTrans ? K : M));
            TS_AUTO_CHECK(ldb == (TransB == cublas::NoTrans ? N : K));
            TS_AUTO_CHECK(ldc == N);

            dim3 blockSize(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);
            TS_AUTO_CHECK(blockSize.x == TRANS_BLOCK_DIM);
            TS_AUTO_CHECK(blockSize.y == TRANS_BLOCK_DIM);

            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y, 1);
            bool transA_bool = TransA == cublas::NoTrans ? false : true;
            bool transB_bool = TransB == cublas::NoTrans ? false : true;

            RUN_KERNEL(gemm_kernel<T>, gridSize, blockSize,
                       transA_bool, transB_bool, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

            return true;
        }

        template<typename T>
        bool math<T>::gemm(
                cublas::Order Order,
                cublas::Transpose TransA,
                cublas::Transpose TransB,
                int M, int N, int K,
                T alpha,
                const T *A, int lda,
                const T *B, int ldb,
                T beta,
                T *C, int ldc) {
            if (Order == cublas::ColMajor) {
                return inline_gemm_row_major<T>(TransB, TransA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
            }
            else {
                return inline_gemm_row_major<T>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
            }
        }

        template<typename T>
        bool math<T>::gemm(cublas::Transpose TransA, cublas::Transpose TransB, int M, int N, int K, T alpha, const T *A,
            const T *B, T beta, T *C) {
            int lda = (TransA == cublas::NoTrans ? K : M);
            int ldb = (TransB == cublas::NoTrans ? N : K);
            int ldc = N;
            return inline_gemm_row_major<T>(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        template<typename T>
        bool math<T>::asum(int N, const T *x, T * out) {
            int grid_size = CUDA_BLOCK(N, CUDA_THREAD_NUM);
            int block_size = CUDA_THREAD_NUM;
            //unsigned int shared_size = block_size * sizeof(T);
            Tensor tmp_tensor(Tensor::InFlow::DEVICE, UINT8, {int32_t(grid_size *sizeof(T))});
            T* tmp_out = tmp_tensor.data<T>();
            RUN_KERNEL(abs_sum_kernel<T>, grid_size, block_size, N, const_cast<T*>(x), tmp_out);
            while (grid_size > CUDA_THREAD_NUM) {
                int len = grid_size;
                grid_size = CUDA_BLOCK(grid_size, CUDA_THREAD_NUM);
                RUN_KERNEL(abs_sum_kernel<T>, grid_size, block_size, len, tmp_out, tmp_out);
            }

            RUN_KERNEL(abs_sum_kernel<T>, 1, grid_size, grid_size, tmp_out, out);
            return true;
        }

        template<typename T>
        bool math<T>::sum(int N, const T *x, T * out) {
            int grid_size = CUDA_BLOCK(N, CUDA_THREAD_NUM);
            int block_size = CUDA_THREAD_NUM;
            //unsigned int shared_size = block_size * sizeof(T);
            Tensor tmp_tensor(Tensor::InFlow::DEVICE, UINT8, {int32_t(grid_size *sizeof(T))});
            T* tmp_out = tmp_tensor.data<T>();
            RUN_KERNEL(sum_kernel<T>, grid_size, block_size, N, const_cast<T*>(x), tmp_out);
            while (grid_size > CUDA_THREAD_NUM) {
                int len = grid_size;
                grid_size = CUDA_BLOCK(grid_size, CUDA_THREAD_NUM);
                RUN_KERNEL(sum_kernel<T>, grid_size, block_size, len, tmp_out, tmp_out);
            }

            RUN_KERNEL(sum_kernel<T>, 1, grid_size, grid_size, tmp_out, out);
            return true;
        }

    }
}
//support "half" on nvidia gpu
#ifdef TS_USE_CUDA_FP16
template class ts::gpu::math<half>;
#endif
template class ts::gpu::math<ts::dtype<ts::FLOAT32>::declare>;
template class ts::gpu::math<ts::dtype<ts::FLOAT64>::declare>;