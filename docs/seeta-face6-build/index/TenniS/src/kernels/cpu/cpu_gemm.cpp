#include <kernels/cpu/cpu_gemm.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif



namespace ts {
    namespace cpu {
        static inline int to_mod_index(const HypeShape &hype, const Shape &coordinate) {
            auto temp = coordinate;
            for (size_t i = 0; i < temp.size(); ++i) {
                temp[i] %= hype.shape(i);
            }
            return hype.to_index(temp);
        }

        template<typename T>
        static inline void cpu_gemm_broadcast_compute_run(const Tensor &C, Tensor &out) {
            HypeShape C_hype(C.sizes());
            ShapeIterator out_iterator(out.sizes());

            auto pC = C.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();
            for (int i = 0; i < ncount; i++) {
                auto &tmpshape = out_iterator.coordinate();
                pout[i] = pC[to_mod_index(C_hype, tmpshape)];
                ++out_iterator;
            }
        }

        template<typename T>
        static void cpu_gemm_compute_run(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                                         float alpha, float beta, bool transA, bool transB, Tensor &out) {
            auto blas_transA = transA ? blas::Trans : blas::NoTrans;
            auto blas_transB = transB ? blas::Trans : blas::NoTrans;

            auto ptr_A = A.data<T>();
            auto ptr_B = B.data<T>();
            auto ptr_C = out.data<T>();

            int M = out.size(0);
            int N = out.size(1);

            // broadcast C to output
            if (!near_zero(beta)) {
                cpu_gemm_broadcast_compute_run<T>(C, out);
            } else {
                beta = 0;
            }

#ifdef TS_USE_CBLAS
            cblas::math<T>::gemm(blas_transA, blas_transB, M, N, K,
                                 (T) alpha, ptr_A, ptr_B, (T) beta, ptr_C);
#else
            cpu::math<T, T>::gemm(blas_transA, blas_transB, M, N, K,
                               (T) alpha, ptr_A, ptr_B, (T) beta, ptr_C);
#endif
        }

        void Gemm::gemm(const Tensor &A, const Tensor &B, const Tensor &C, int K,
                  float alpha, float beta, bool transA, bool transB, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_gemm_compute_run<TYPE>(A, B, C, K, alpha, beta, transA, transB, out); break; }
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
using namespace cpu;
TS_REGISTER_OPERATOR(Gemm, CPU, name::layer::gemm())
