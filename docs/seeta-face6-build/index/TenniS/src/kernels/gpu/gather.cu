#include <backend/base/base_gather.h>

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
        class Gather : public OperatorOnGPU<base::Gather> {
        public:
            using self = Gather;
            using supper = OperatorOnGPU<base::Gather>;

            Gather() = default;

            void gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) override;
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
        static __global__ void gpu_gather_tensor_kernel(
                int count, const T*x, T*out,
                const int32_t *indices, int x_slice,
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
            int in_index = (n * x_slice + indices[s]) * width + w;

            out[out_index] = x[in_index];
        }

        template<typename T>
        static inline void gpu_gather_tensor_compute_run(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            auto &x_shape = x.sizes();
            auto &out_shape = out.sizes();
            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1, std::multiplies<int32_t>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1, std::multiplies<int32_t>());

            auto x_slice = x_shape[axis];
            auto out_slice = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int32_t>());

            auto count = out.count();

            RUN_KERNEL(gpu_gather_tensor_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, x.data<T>(), out.data<T>(),
                       indices.data<int32_t>(), x_slice,
                       number, out_slice, width,
                       out_slice * width);
        }


        void Gather::gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_gather_tensor_compute_run<TYPE>(x, indices.view(Tensor::InFlow::DEVICE), axis, out); break; }
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
TS_REGISTER_OPERATOR(Gather, GPU, name::layer::gather())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Gather, GPU, name::layer::gather())
#endif
