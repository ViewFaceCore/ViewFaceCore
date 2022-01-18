//
// Created by kier on 19-4-17.
//

#ifndef TENSORSTACK_KERNELS_GPU_DCN_UTILS_H
#define TENSORSTACK_KERNELS_GPU_DCN_UTILS_H

#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "runtime/runtime.h"
#include <cstdint>
#include <climits>
#include "core/device_context.h"
#include "kernels/gpu/cuda_context.h"

#include <algorithm>

// #define __HIP_PLATFORM_HCC__

namespace ts {
    namespace dcn {
        inline Tensor empty(const Tensor::Prototype &proto, const MemoryDevice &device) {
            return Tensor(RuntimeContext::FlowMemory(), proto, device);
        }

        inline Tensor ones(const Tensor::Prototype &proto, const MemoryDevice &device) {
            Tensor tensor = empty(proto, device);
            Tensor one = tensor::build(proto.dtype(), 1);
            memset(tensor.data(), tensor.device(), tensor.count() * tensor.proto().type_bytes(),
                   one.data(), one.device(), one.count() * one.proto().type_bytes());
            return std::move(tensor);
        }

        inline Tensor zeros(const Tensor::Prototype &proto, const MemoryDevice &device) {
            Tensor tensor = empty(proto, device);
            Tensor one = tensor::build(proto.dtype(), 0);
            memset(tensor.data(), tensor.device(), tensor.count() * tensor.proto().type_bytes(),
                   one.data(), one.device(), one.count() * one.proto().type_bytes());
            return std::move(tensor);
        }

        inline Tensor empty(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return empty(Tensor::Prototype(dtype, shape), device);
        }

        inline Tensor ones(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return ones(Tensor::Prototype(dtype, shape), device);
        }

        inline Tensor zeros(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return ones(Tensor::Prototype(dtype, shape), device);
        }

        inline void *CudaMalloc(const MemoryDevice &device, size_t size, SyncMemory &buffer) {
            buffer = RuntimeContext::FlowMemory()->alloc(device, size);
            return buffer.data();
        }

        inline void
        adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb,
                       int64_t *ldc) {
            int transa_ = ((transa == 't') || (transa == 'T'));
            int transb_ = ((transb == 't') || (transb == 'T'));

            // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
            // requires (even if the value won't be used).
            if (n <= 1)
                *ldc = std::max<int64_t>(m, 1);

            if (transa_) {
                if (m <= 1)
                    *lda = std::max<int64_t>(k, 1);
            } else {
                if (k <= 1)
                    *lda = std::max<int64_t>(m, 1);
            }

            if (transb_) {
                if (k <= 1)
                    *ldb = std::max<int64_t>(n, 1);
            } else {
                if (n <= 1)
                    *ldb = std::max<int64_t>(k, 1);
            }

        }

        static cublasOperation_t convertTransToCublasOperation(char trans) {
            if (trans == 't') return CUBLAS_OP_T;
            else if (trans == 'n') return CUBLAS_OP_N;
            else if (trans == 'c') return CUBLAS_OP_C;
            else {
                TS_LOG_ERROR("trans must be one of: t, n, c") << eject;
                return CUBLAS_OP_T;
            }
        }

        static void CudaBlas_SgemmStridedBatched(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                                 float alpha, const float *a, int64_t lda, int64_t strideA,
                                                 const float *b,
                                                 int64_t ldb, int64_t strideB,
                                                 float beta, float *c, int64_t ldc, int64_t strideC,
                                                 int64_t batchCount) {
            if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX) ||
                (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX)) {
                TS_LOG_ERROR << "Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
                                "with the bound [val] <= " << INT_MAX << eject;
            }

            adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
            cublasOperation_t opa = convertTransToCublasOperation(transa);
            cublasOperation_t opb = convertTransToCublasOperation(transb);

            auto &context = ctx::ref<DeviceContext>();
            auto *handle = reinterpret_cast<CUDAContextHandle *>(context.handle);

            // cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
            // cublasSetStream(handle, THCState_getCurrentStream(state));
            if (CUBLAS_STATUS_SUCCESS != cublasSgemmStridedBatched(handle->cublas_handle(),
                                                                   opa, opb, (int) m, (int) n, (int) k,
                                                                   &alpha, a, (int) lda, strideA, b, (int) ldb, strideB,
                                                                   &beta, c, (int) ldc, strideC,
                                                                   (int) batchCount)) {
                TS_LOG_ERROR << "Run cublasSgemmStridedBatched failed." << eject;
            }
        }

        static void CudaBlas_SgemmBatched(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                          float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                                          float beta, float *c[], int64_t ldc, int64_t batchCount) {
            if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) ||
                (lda >= INT_MAX) || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX)) {
                TS_LOG_ERROR << "Cublas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
                                "with the bound [val] <= " << INT_MAX << eject;
            }

#ifdef __HIP_PLATFORM_HCC__

            const int64_t stridea = (transa == 'N' || transa == 'n') ? lda * k : lda * n;
            const int64_t strideb = (transb == 'N' || transb == 'n') ? ldb * n : ldb * k;
            const int64_t stridec = ldc * n;

            CudaBlas_SgemmStridedBatched(transa, transb, m, n, k,
                                         alpha, *a, lda, stridea, *b, ldb, strideb, beta, *c, ldc,
                                         stridec, batchCount);

#else

            adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
            cublasOperation_t opa = convertTransToCublasOperation(transa);
            cublasOperation_t opb = convertTransToCublasOperation(transb);

            auto &context = ctx::ref<DeviceContext>();
            auto *handle = reinterpret_cast<CUDAContextHandle *>(context.handle);

            // cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
            // cublasSetStream(handle, THCState_getCurrentStream(state));
            if (CUBLAS_STATUS_SUCCESS != cublasSgemmBatched(handle->cublas_handle(),
                                                            opa, opb, (int) m, (int) n, (int) k,
                                                            &alpha, a, (int) lda, b, (int) ldb, &beta, c, (int) ldc,
                                                            (int) batchCount)) {
                TS_LOG_ERROR << "Run cublasSgemmBatched failed." << eject;
            }
#endif
        }
    }
}

#endif //TENSORSTACK_KERNELS_GPU_DCN_UTILS_H
