#include <backend/base/base_concat.h>

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
        class Concat : public OperatorOnGPU<base::Concat> {
        public:
            using self = Concat;
            using supper = OperatorOnGPU<base::Concat>;

            Concat() = default;

            void concat(const std::vector<Tensor> &x, int dim, Tensor &out) override;
        };

        /**
         *
         * @tparam T
         * @param count equals to number * slice * width
         * @param N number of x
         * @param x list of pointer of input
         * @param out pointer of output
         * @param slice_map list of int, map slice index to x index
         * @param number
         * @param slice
         * @param width
         * @param number_step = slice * width
         */
        template<typename T>
        static __global__ void gpu_concat_tensor_kernel(
                int count, const T **x, T *out,
                const int32_t *slice_size,
                const int32_t *slice_shift,
                const int32_t *slice_map,
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

            auto i = slice_map[s];
            auto in = x[i];

            slice = slice_size[i];
            s = s - slice_shift[i];

            int in_index = (n * slice + s) * width + w;

            out[out_index] = in[in_index];
        }

        template<typename T>
        static inline void gpu_concat_tensor_compute_run(const std::vector<Tensor> &x, int axis, Tensor &out) {
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
            Tensor slice_size_cpu(Tensor::InFlow::HOST, INT32, {int32_t(N)});
            for (size_t i = 0; i < N; ++i) {
                slice_size_cpu.data<int32_t>(i) = x[i].size(axis);
            }
            Tensor slice_size_gpu = slice_size_cpu.view(Tensor::InFlow::DEVICE);
            Tensor slice_shift_cpu(Tensor::InFlow::HOST, INT32, {int32_t(N)});
            for (size_t i = 0; i < N; ++i) {
                if (i) {
                    slice_shift_cpu.data<int32_t>(i) = slice_shift_cpu.data<int32_t>(i - 1) + x[i - 1].size(axis);
                } else {
                    slice_shift_cpu.data<int32_t>(i) = 0;
                }
            }
            Tensor slice_shift_gpu = slice_shift_cpu.view(Tensor::InFlow::DEVICE);
            Tensor slice_map_cpu(Tensor::InFlow::HOST, INT32, {slice});
            size_t slice_map_i = 0;
            for (size_t i = 0; i < N; ++i) {
                auto x_i_axis = x[i].size(axis);
                for (int j = 0; j < x_i_axis; ++j) {
                    slice_map_cpu.data<int32_t>(slice_map_i) = int32_t(i);
                    ++slice_map_i;
                }
            }
            Tensor slice_map_gpu = slice_map_cpu.view(Tensor::InFlow::DEVICE);

            RUN_KERNEL(gpu_concat_tensor_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, x_data_gpu.data<const T *>(), out.data<T>(),
                       slice_size_gpu.data<int32_t>(),
                       slice_shift_gpu.data<int32_t>(),
                       slice_map_gpu.data<int32_t>(),
                       number, slice, width,
                       slice * width);
        }


        void Concat::concat(const std::vector<Tensor> &x, int dim, Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (type_bytes(dtype)) {
#define DECLARE_COMPUTE_RUN(WIDTH, TYPE) case WIDTH: { gpu_concat_tensor_compute_run<TYPE>(x, dim, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype)
                                 << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Concat, GPU, name::layer::concat())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Concat, GPU, name::layer::concat())
#endif
