//
// Created by kier on 19-4-17.
//

#ifndef TENSORSTACK_KERNELS_CPU_DCN_UTILS_H
#define TENSORSTACK_KERNELS_CPU_DCN_UTILS_H

#include "core/tensor.h"
#include "core/tensor_builder.h"
#include "runtime/runtime.h"
#include <cstdint>
#include <climits>
#include "core/device_context.h"

#include <algorithm>
#include <kernels/common/blas.h>
#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#else
#include <kernels/cpu/math_cpu.h>
#endif

namespace ts {
    namespace dcn {
        namespace cpu {
            inline Tensor empty(const Tensor::Prototype &proto) {
                return Tensor(Tensor::InFlow::HOST, proto);
            }

            inline Tensor ones(const Tensor::Prototype &proto) {
                Tensor tensor = empty(proto);
                Tensor one = tensor::build(proto.dtype(), 1);
                memset(tensor.data(), tensor.device(), tensor.count() * tensor.proto().type_bytes(),
                       one.data(), one.device(), one.count() * one.proto().type_bytes());
                return std::move(tensor);
            }

            inline Tensor zeros(const Tensor::Prototype &proto) {
                Tensor tensor = empty(proto);
                Tensor one = tensor::build(proto.dtype(), 0);
                memset(tensor.data(), tensor.device(), tensor.count() * tensor.proto().type_bytes(),
                       one.data(), one.device(), one.count() * one.proto().type_bytes());
                return std::move(tensor);
            }

            inline Tensor empty(DTYPE dtype, const Shape &shape) {
                return empty(Tensor::Prototype(dtype, shape));
            }

            inline Tensor ones(DTYPE dtype, const Shape &shape) {
                return ones(Tensor::Prototype(dtype, shape));
            }

            inline Tensor zeros(DTYPE dtype, const Shape &shape) {
                return zeros(Tensor::Prototype(dtype, shape));
            }


            inline void *CPUMalloc(const MemoryDevice &device, size_t size, SyncMemory &buffer) {
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

            static blas::Transpose convertTransToCublasOperation(char trans) {
                if (trans == 't') return blas::Trans;
                else if (trans == 'n') return blas::NoTrans;
                else {
                    TS_LOG_ERROR("trans must be one of: t, n, c") << eject;
                    return blas::Trans;
                }
            }

            void CBlas_SgemmBatched(blas::Transpose transa, blas::Transpose transb, int64_t m, int64_t n, int64_t k,
                                    float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                                    float beta, float *c[], int64_t ldc, int64_t batchCount) {
				for (int i = 0; i < batchCount; ++i) {
#ifdef TS_USE_CBLAS
					cblas::math<float>::gemm(blas::ColMajor, transa, transb,
						int(m), int(n), int(k),
						alpha, a[i], int(lda),
						b[i], int(ldb),
						beta, c[i], int(ldc));
#else
					ts::cpu::math<float, float>::gemm(blas::ColMajor, transa, transb,
						int(m), int(n), int(k),
						alpha, a[i], int(lda),
						b[i], int(ldb),
						beta, c[i], int(ldc));
#endif
				}
            }

            static void CBlas_SgemmBatched(char transa, char transb, int64_t m, int64_t n, int64_t k,
                                           float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                                           float beta, float *c[], int64_t ldc, int64_t batchCount) {
                if ((m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) ||
                    (lda >= INT_MAX) || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX)) {
                    TS_LOG_ERROR << "CBlas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
                                    "with the bound [val] <= " << INT_MAX << eject;
                }

                adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
                blas::Transpose opa = convertTransToCublasOperation(transa);
                blas::Transpose opb = convertTransToCublasOperation(transb);

                CBlas_SgemmBatched(opa, opb, (int) m, (int) n, (int) k,
                                   alpha, a, (int) lda, b, (int) ldb, beta, c, (int) ldc,
                                   (int) batchCount);
            }
        }
    }
}



#endif //TENSORSTACK_KERNELS_CPU_DCN_UTILS_H
