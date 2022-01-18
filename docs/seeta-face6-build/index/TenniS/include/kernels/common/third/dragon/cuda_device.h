//
// Created by kier on 2019/9/12.
//

#ifndef TENSORSTACK_THIRD_DRAGON_CUDA_DEVICE_H
#define TENSORSTACK_THIRD_DRAGON_CUDA_DEVICE_H

#ifdef TS_USE_CUDA

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>

namespace ts {
    namespace dragon {
        /*!
         * The number of cuda threads to use.
         *
         * We set it to 1024 which would work for compute capability 2.x.
         *
         * Set it to 512 if using compute capability 1.x.
         */
        const int CUDA_THREADS = 1024;

        /*!
         * The maximum number of blocks to use in the default kernel call.
         *
         * We set it to 65535 which would work for compute capability 2.x,
         * where 65536 is the limit.
         */
        const int CUDA_MAX_BLOCKS = 65535;

        inline int CUDA_BLOCKS(const int N) {
            return std::max(
                    std::min(
                            (N + CUDA_THREADS - 1) / CUDA_THREADS,
                            CUDA_MAX_BLOCKS
                    ), 1);
        }
    }
}

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < n; i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) \
        << "\n" << cudaGetErrorString(error) << eject; \
  } while (0)

#endif

#endif //TENSORSTACK_THIRD_DRAGON_CUDA_DEVICE_H
