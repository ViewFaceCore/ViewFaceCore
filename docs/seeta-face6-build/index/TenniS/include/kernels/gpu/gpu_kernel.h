//
// Created by kier on 2020/1/20.
//

#ifndef TENNIS_KERNELS_GPU_GPU_KERNEL_H
#define TENNIS_KERNELS_GPU_GPU_KERNEL_H

#include <cuda_runtime.h>

#include "gpu_helper.h"
#include "utils/log.h"

#define RUN_KERNEL(kernel, grid, block, ...) \
do { \
    auto stream = ts::gpu::get_cuda_stream_on_context(); \
    kernel<<<(grid), (block), 0, stream>>> (__VA_ARGS__); \
    auto errcode = cudaGetLastError(); \
    if (errcode) TS_LOG_ERROR << "Got cuda error(" << errcode << ") " << cudaGetErrorString(errcode) << ts::eject; \
} while (false)

#define RUN_KERNEL_STREAM(kernel, grid, block, shared, stream, ...) \
do { \
    kernel<<<(grid), (block), (shared), (stream)>>> (__VA_ARGS__); \
    auto errcode = cudaGetLastError(); \
    if (errcode) TS_LOG_ERROR << "Got cuda error(" << errcode << ") " << cudaGetErrorString(errcode) << ts::eject; \
} while (false)

#define CUDA_CHECK_LAST_ERROR \
{ \
    auto errcode = cudaGetLastError(); \
    if (errcode) TS_LOG_ERROR << "Got cuda error(" << errcode << ") " << cudaGetErrorString(errcode) << ts::eject; \
}

#endif //TENNIS_KERNELS_GPU_GPU_KERNEL_H
