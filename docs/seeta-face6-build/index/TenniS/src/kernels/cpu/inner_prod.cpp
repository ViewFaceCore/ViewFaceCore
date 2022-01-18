#include <kernels/cpu/inner_prod.h>
#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>
#include <frontend/intime.h>

#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif



namespace ts {
    namespace cpu {
        template<typename T>
        static void cpu_inner_prod_compute_run(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out, Stack &stack, bool kernel_packed) {
            const Shape &lhs_shape = lhs.sizes();
            const Shape &rhs_shape = rhs.sizes();
            // const Shape &out_shape = out.sizes();

            T *pdst = out.data<T>();
            auto N = transpose ? rhs_shape[0] : rhs_shape[1];

#ifdef TS_USE_CBLAS
            const T *psrc = lhs.data<T>();
            const T *pdot = rhs.data<T>();
            auto rhs_transpose = transpose ? blas::Trans : blas::NoTrans;
            cblas::math<T>::gemm(blas::NoTrans, rhs_transpose, lhs_shape[0], N, lhs_shape[1],
                                 (T) 1, psrc, pdot, (T) 0, pdst);
#else
            if (transpose && kernel_packed) {
                 TS_LOG_ERROR << "What a Terrible Failure: dealing transpose weights without transpose support, because supporting pack" << eject;
            }

            auto no_trasposed_rhs = rhs;
            if (transpose) {
                no_trasposed_rhs = intime::transpose(rhs, {1, 0});
            }
            auto rhs_data = no_trasposed_rhs.data<T>();

            Tensor lhs_packed = stack.make(lhs.dtype(), lhs_shape, MemoryDevice(CPU));
            Tensor rhs_packed = stack.make(rhs.dtype(), rhs_shape, MemoryDevice(CPU));
            
            cpu::math<T, T>::gemm(lhs_shape[0], N, lhs_shape[1], (T)1, lhs.data<T>(), lhs_packed.data<T>(),
                                  rhs_data, rhs_packed.data<T>(), T(0), pdst, true, !kernel_packed);
            //cpu::math<T, T>::gemm(blas::NoTrans, rhs_transpose, lhs_shape[0], N, lhs_shape[1],
            //                   (T) 1, psrc, pdot, (T) 0, pdst);
#endif
        }

        void InnerProd::inner_prod(const Tensor &lhs, const Tensor &rhs, bool transpose, Tensor &out, Stack &stack, bool kernel_packed) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_inner_prod_compute_run<TYPE>(lhs, rhs, transpose, out, stack, kernel_packed); break; }
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
TS_REGISTER_OPERATOR(InnerProd, CPU, name::layer::inner_prod())
