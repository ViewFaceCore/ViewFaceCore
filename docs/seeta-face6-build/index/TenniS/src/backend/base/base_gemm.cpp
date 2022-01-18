//
// Created by kier on 2019/3/5.
//

#include <backend/base/base_gemm.h>

#include "backend/base/base_gemm.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        static inline void front_append_ones(Shape &shape, int count) {
            Shape ones(count, 1);
            shape.insert(shape.begin(), ones.begin(), ones.end());
        }

        static void infer_gemm(const Tensor &A, const Tensor &B, const Tensor &C,
             float alpha, float beta, bool transA, bool transB,
             int &K,
             Tensor::Prototype &out, Shape &adjusted_C_shape) {
            if(A.dims() != 2) TS_LOG_ERROR << "A.dims() expect equal 2, got " << A.dims() << eject;
            if(B.dims() != 2) TS_LOG_ERROR << "A.dims() expect equal 2, got " << B.dims() << eject;
            if(C.dims() > 2) TS_LOG_ERROR << "A.dims() expect less 2, got " << C.dims() << eject;

            TS_AUTO_CHECK(A.dtype() == B.dtype());

            TS_AUTO_CHECK(beta == 0 || C.dtype() == A.dtype());

            int M, A_K, B_K, N;
            if (transA) {
                A_K = A.size(0);
                M = A.size(1);
            } else {
                M = A.size(0);
                A_K = A.size(1);
            }
            if (transB) {
                N = B.size(0);
                B_K = B.size(1);
            } else {
                B_K = B.size(0);
                N = B.size(1);
            }

            if(A_K != B_K) {
                TS_LOG_ERROR << "Can not gemm("
                    << "A=" << to_string(A.sizes()) << ", "
                    << "B=" <<  to_string(B.sizes()) << ", "
                    << "C=" <<  to_string(C.sizes()) << ", "
                    << "alpha=" <<  alpha << ", "
                    << "beta=" <<  beta << ", "
                    << std::boolalpha
                    << "transA=" <<  transA << ", "
                    << "transB=" <<  transB << ")"
                    << eject;
            }

            auto C_shape = C.sizes();
            if (C_shape.size() < 2) {
                front_append_ones(C_shape, int(2 - C_shape.size()));
            }

            auto gemm_out = Tensor::Prototype(A.dtype(), {M, N});

            if ((C_shape[0] != 1 && C_shape[0] != M) || (C_shape[1] != 1 && C_shape[1] != N)) {
                TS_LOG_ERROR << "Can not broadcast " << to_string(C.sizes()) << " to " << to_string(out.sizes()) << eject;
            }

            K = A_K;
            out = gemm_out;
            adjusted_C_shape = C_shape;
        }

        Gemm::Gemm() {
            field(name::alpha, OPTIONAL, tensor::from<float>(1.0f));
            field(name::beta, OPTIONAL, tensor::from<float>(1.0f));
            field(name::transA, OPTIONAL, tensor::from<bool>(false));
            field(name::transB, OPTIONAL, tensor::from<bool>(false));
        }

        void Gemm::init() {
            supper::init();
            m_alpha = tensor::to_float(get(name::alpha));
            m_beta = tensor::to_float(get(name::beta));
            m_transA = tensor::to_bool(get(name::transA));
            m_transB = tensor::to_bool(get(name::transB));
        }

        int Gemm::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto &A = stack[0];
            auto &B = stack[1];
            auto &C = stack[2];

            output.resize(1);

            int K;
            Shape adjusted_C_shape;
            infer_gemm(A, B, C, m_alpha, m_beta, m_transA, m_transB, K, output[0], adjusted_C_shape);

            return 1;
        }

        int Gemm::run(Stack &stack) {
            TS_AUTO_CHECK(stack.size() == 3);

            auto memory_device = running_memory_device();

            auto A = stack[0].view(memory_device);
            auto B = stack[1].view(memory_device);
            auto C = stack[2].view(memory_device);

            int K;
            Tensor::Prototype output_proto;
            Shape adjusted_C_shape;
            infer_gemm(A, B, C, m_alpha, m_beta, m_transA, m_transB, K, output_proto, adjusted_C_shape);

            C = C.reshape(adjusted_C_shape);

            auto &out = *stack.push(output_proto, memory_device);

            gemm(A, B, C, K, m_alpha, m_beta, m_transA, m_transB, out);

            return 1;
        }
    }

}

