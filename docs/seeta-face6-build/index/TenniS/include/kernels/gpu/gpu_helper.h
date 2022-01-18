//
// Created by kier on 19-3-30.
//

#ifndef TENSORSTACK_GPU_HELPER_H
#define TENSORSTACK_GPU_HELPER_H


#include <core/sync/sync_memory.h>
#include <core/sync/sync_controller.h>
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ts {
    namespace gpu {
        using CpuBlock = std::pair<void *, int>;
        /**
         * convert cpu memory to gpu, by using single gpu memory block
         * @param [in] controller controller to memory alloc
         * @param [in] device device contain memory
         * @param [in] cpu list of cpu memory
         * @param [out] gpu pointer containing converted gpu memory pointer
         * @return the memory contain all `gpu` memory
         */
        SyncMemory convert_block_to_gpu(
                SyncMemoryController::shared controller,
                const MemoryDevice &device,
                const std::vector<CpuBlock> &cpu,
                const std::vector<void **> &gpu);

        struct GpuHypeShape {
            int32_t dims = 0;
            int32_t *shape = nullptr;   // each shape size, [dim_i]_{i=0}^{dims}
            int32_t *weights = nullptr; // each weight, [weight_i]_{i=0}^{dims}, and weight_i = \sum_{j=i}^{dims} dim_i
        };

        // std::pair<SyncMemory, GpuHypeShape> MakeGPUHypeShape(const MemoryDevice &device, const Shape &shape);
        std::pair<SyncMemory, std::vector<GpuHypeShape>> MakeGPUHypeShape(const MemoryDevice &device, const std::vector<Shape> &shape);
        std::pair<SyncMemory, std::vector<GpuHypeShape>> MakeGPUHypeShape(const MemoryDevice &device, const std::vector<Shape> &shape,
                                                                          const std::vector<CpuBlock> &cpu,
                                                                          const std::vector<void **> &gpu);


        /**
        * get cuda stream on current context
        * @return  cuda stream on current context
        */
        cudaStream_t get_cuda_stream_on_context();
    }
}


#endif //TENSORSTACK_GPU_HELPER_H
