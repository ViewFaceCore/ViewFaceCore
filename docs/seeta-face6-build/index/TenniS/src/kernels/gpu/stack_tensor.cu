#include <backend/base/base_stack_tensor.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <kernels/gpu/operator_on_gpu.h>
#include "kernels/gpu/cuda_context.h"
#include "utils/ctxmgr_lite.h"
#include "core/device_context.h"

#ifdef TS_USE_CUBLAS
#include "kernels/gpu/math_cublas.h"
#else
#include "kernels/gpu/math_gpu.h"
#endif

#include "kernels/gpu/gpu_kernel.h"

#include "global/operator_factory.h"
#include "global/fp16_operator_factory.h"
#include "backend/name.h"
#include <numeric>


namespace ts {
    namespace gpu {
        class StackTensor : public OperatorOnGPU<base::StackTensor> {
        public:
            using self = StackTensor;
            using supper = OperatorOnGPU<base::StackTensor>;

            StackTensor() = default;

            void stack_tensor(const std::vector<Tensor> &x, int axis, Tensor &out) override;
        };

        /**
         *
         * @tparam T
         * @param count equals to number * slice * width
         * @param N number of x
         * @param x list of pointer of input
         * @param out pointer of output
         * @param number
         * @param slice
         * @param width
         * @param number_step = slice * width
         */
        template<typename T>
        static __global__ void gpu_stack_tensor_kernel(
                int count, const T**x, T*out,
                int number, int slice, int width,
                int number_step) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            auto slice_step = width;

            int out_index = index;

            auto n = index / number_step;
            index %= number_step;
            auto s = index / slice_step;
            index %= slice_step;
            auto w = index;
            int in_index = n * width + w;

            out[out_index] = x[s][in_index];
        }

        template<typename T>
        static inline void gpu_stack_tensor_compute_run(const std::vector<Tensor> &x, int axis, Tensor &out) {
            auto &out_shape = out.sizes();
            auto number = std::accumulate(out_shape.begin(), out_shape.begin() + axis, 1, std::multiplies<int32_t>());
            auto slice = out_shape[axis];
            auto width = std::accumulate(out_shape.begin() + axis + 1, out_shape.end(), 1, std::multiplies<int32_t>());

            auto N = x.size();

            auto count = out.count();
            Tensor x_data_cpu(Tensor::InFlow::HOST, PTR, {int32_t(N)});
            for (size_t i = 0; i < N; ++i) {
                x_data_cpu.data<const T *>(i) = x[i].data<T>();
            }
            Tensor x_data_gpu = x_data_cpu.view(Tensor::InFlow::DEVICE);

            RUN_KERNEL(gpu_stack_tensor_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, x_data_gpu.data<const T *>(), out.data<T>(),
                       number, slice, width,
                       slice * width);
        }


        void StackTensor::stack_tensor(const std::vector<Tensor> &x, int axis, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_stack_tensor_compute_run<TYPE>(x, axis, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
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
TS_REGISTER_OPERATOR(StackTensor, GPU, name::layer::stack())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(StackTensor, GPU, name::layer::stack())
#endif
