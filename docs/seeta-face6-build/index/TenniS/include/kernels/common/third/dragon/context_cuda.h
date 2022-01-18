//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H
#define TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H

#include <cstdint>
#include <cstdlib>

#include "context.h"

#ifdef TS_USE_CUDA
#include "cuda_device.h"
#include "kernels/gpu/gpu_helper.h"
#include "kernels/gpu/memory_gpu.h"
#endif

namespace ts {
    namespace dragon {
        class Workspace;
        class CPUContext;

#ifdef TS_USE_CUDA

        template<typename T, typename TD, typename FD>
        class CUDAMemory {
        public:
            void static Copy(size_t count, T *dst, const T *src, cudaStream_t stream);
        };

        class CUDAContext : public BaseContext {
        public:
            using self = CUDAContext;
            using supper = BaseContext;

            CUDAContext(Workspace *ws) : supper(ws) {}

            template<typename T, typename TD, typename FD>
            void Copy(size_t count, T *dst, const T *src) {
                CUDAMemory<T, TD, FD>::Copy(count, dst, src, this->cuda_stream());
            }

            cudaStream_t cuda_stream() {
                return gpu::get_cuda_stream_on_context();
            }
        };

        template<typename T>
        class CUDAMemory<T, CPUContext, CPUContext> {
        public:
            void static Copy(size_t count, T *dst, const T *src, cudaStream_t) {
                std::memcpy(dst, src, count * sizeof(T));
            }
        };

        template<typename T>
        class CUDAMemory<T, CUDAContext, CUDAContext> {
        public:
            void static Copy(size_t count, T *dst, const T *src, cudaStream_t stream) {
                cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice, stream);
            }
        };

#else
        class CUDAContext : public BaseContext {};
#endif
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_CONTEXT_CUDA_H
