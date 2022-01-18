//
// Created by kier on 2019/3/6.
//

#include "kernels/gpu/gatherv2.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "kernels/gpu/gpu_kernel.h"
#include <numeric>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"

namespace ts {
    namespace gpu {
        static __global__ void gpu_gatherv2_kernel(int count, const char * x_data, const int * indices_data, char * out_data, 
                                                   int axis, int bytes, int width_bytes, GpuHypeShape c_shape) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= count) return;

            int in_index = 0;
            auto in_weight_it = c_shape.weights + 1;
            int curindex = 0;
            
            for (int  k=0; k < axis; k++) {
                curindex = indices_data[index * axis + k];

                if(k >= c_shape.dims -1) {
                    in_index += curindex;
                }else {
                    in_index += *in_weight_it  * curindex;
                    in_weight_it++;
                }
            }

            auto src_ptr = x_data + in_index * bytes;
            auto dst_ptr = out_data + index * width_bytes;
            ::memcpy((void *)dst_ptr, (void *)src_ptr,width_bytes);
        }


        void GatherV2::gather(const Tensor &x, const Tensor &indices, Tensor &out) {
            auto memcpy_handler = HardConverter::Query(out.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy_handler != nullptr);
            auto device_id = out.device().id();

            auto &x_shape = x.sizes();
            auto &i_shape = indices.sizes();

            int axis = i_shape[i_shape.size() - 1];

            auto number = std::accumulate(i_shape.begin(), i_shape.end() - 1, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis, x_shape.end(), 1, std::multiplies<int>());

            auto gpu_hype_shape = MakeGPUHypeShape(x.device(), {x_shape});
            auto &x_hype_shape = gpu_hype_shape.second[0];

            auto bytes = x.proto().type_bytes();
            auto width_bytes = width * bytes;

            auto x_data = x.data<char>();
            auto out_data = out.data<char>();
            auto indices_data = indices.data<int32_t>();

            RUN_KERNEL(gpu_gatherv2_kernel, CUDA_BLOCK(number, CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       number, x_data, indices_data, out_data, axis, bytes, width_bytes, x_hype_shape);

        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(GatherV2, GPU, name::layer::gatherv2())
