//
// Created by kier on 2019/4/8.
//

#include "global/operator_factory.h"
#include "backend/name.h"

#include "core/tensor_iterator.h"

#include <numeric>
#include <utils/ctxmgr.h>

#include "kernels/gpu/operator_on_gpu.h"
#include "backend/base/base_strided_slice.h"

#include "kernels/gpu/gpu_kernel.h"
#include "device_launch_parameters.h"


namespace ts {
    namespace gpu {
        class StridedSlice : public OperatorOnGPU<base::StridedSlice> {
        public:
            using self = StridedSlice;
            using supper = OperatorOnGPU<base::StridedSlice>;

            void strided_slice(
                    const Tensor &x,
                    const std::vector<int> &begin,
                    const std::vector<int> &end,
                    const std::vector<int> &stride,
                    Tensor &out) override;
        };

        static void insert_back_zeros(std::vector<int> &arr, size_t count) {
            std::vector<int> zeros(count, 0);
            arr.insert(arr.end(), zeros.begin(), zeros.end());
        }

        static void insert_back_ones(std::vector<int> &arr, size_t count) {
            std::vector<int> ones(count, 1);
            arr.insert(arr.end(), ones.begin(), ones.end());
        }

        class GPUSlice {
        public:
            size_t size;
            int32_t *begin;
            int32_t *stride;
        };

        class GPUSliceTensor {
        public:
            Tensor buffer;
            GPUSlice slice;
        };

        static GPUSliceTensor build_slice(const std::vector<int32_t> &begin,
                                          const std::vector<int32_t> &stride) {
            Tensor cpu_slice(Tensor::InFlow::HOST, INT32, {int(begin.size() * 2)});

            std::memcpy(cpu_slice.data<int32_t>(), begin.data(), sizeof(int32_t) * begin.size());
            std::memcpy(cpu_slice.data<int32_t>() + begin.size(), stride.data(), sizeof(int32_t) * stride.size());

            GPUSliceTensor gpu_slice;
            gpu_slice.buffer = cpu_slice.view(Tensor::InFlow::DEVICE);
            gpu_slice.slice.begin = gpu_slice.buffer.data<int32_t>();
            gpu_slice.slice.stride = gpu_slice.buffer.data<int32_t>() + begin.size();

            return gpu_slice;
        }

        template<typename T>
        static __global__ void gpu_stride_slice_kernel(int count, const T *x, T *out,
                                                       GpuHypeShape x_shape, GpuHypeShape out_shape, GPUSlice slice) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int out_index = index;
            int in_index = 0;

            auto out_weight_it = out_shape.weights + 1;
            auto in_weight_it = x_shape.weights + 1;
            /* ============================================ */
            auto stride_it = slice.stride;
            auto begin_it = slice.begin;
            /* ============================================ */

            for (int times = out_shape.dims - 1; times; --times) {
                auto coord = index / *out_weight_it;
                /* ============================================ */
                coord = coord * *stride_it + *begin_it;
                ++stride_it;
                ++begin_it;
                /* ============================================ */
                in_index += coord * *in_weight_it;
                index %= *out_weight_it;
                ++out_weight_it;
                ++in_weight_it;
            }
            auto coord = index;
            /* ============================================ */
            coord = coord * *stride_it + *begin_it;
            /* ============================================ */
            in_index += coord;

            /* ++++++++++++++++++++++++++++++++++++++++++++ */
            out[out_index] = x[in_index];
        }

        template<typename T>
        static void gpu_compute_strided_slice(const ts::Tensor &x, const std::vector<int> &begin,
                                              const std::vector<int> &end, const std::vector<int> &stride,
                                              ts::Tensor &out) {
            auto &x_shape = x.sizes();

            // map begin and stride
            auto fixed_begin = begin;
            insert_back_zeros(fixed_begin, x_shape.size() - begin.size());
            auto fixed_stride = stride;
            insert_back_ones(fixed_stride, x_shape.size() - stride.size());
            auto gpu_slice = build_slice(fixed_begin, fixed_stride);

            // map x, y shape
            auto gpu_hype_shape = MakeGPUHypeShape(x.device(), {x.sizes(), out.sizes()});
            auto &x_hype_shape = gpu_hype_shape.second[0];
            auto &out_hype_shape = gpu_hype_shape.second[1];
            auto count = out.count();

            auto in_data = x.data<T>();
            auto out_data = out.data<T>();

            RUN_KERNEL(gpu_stride_slice_kernel<T>, CUDA_BLOCK(count, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       count, in_data, out_data,
                       x_hype_shape, out_hype_shape, gpu_slice.slice);
        }

        void StridedSlice::strided_slice(const ts::Tensor &x, const std::vector<int> &begin,
                                         const std::vector<int> &end, const std::vector<int> &stride,
                                         ts::Tensor &out) {
            auto type_bytes = out.proto().type_bytes();
            switch (type_bytes) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { gpu_compute_strided_slice<TYPE>(x, begin, end, stride, out); break; }
                DECLARE_COMPUTE_RUN(1, uint8_t)
                DECLARE_COMPUTE_RUN(2, uint16_t)
                DECLARE_COMPUTE_RUN(4, uint32_t)
                DECLARE_COMPUTE_RUN(8, uint64_t)
#undef DECLARE_COMPUTE_RUN
                default: {
                    auto dtype = out.dtype();
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
TS_REGISTER_OPERATOR(StridedSlice, GPU, name::layer::strided_slice())
