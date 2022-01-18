//
// Created by kier on 2018/11/2.
//

#ifndef TENSORSTACK_KERNELS_GPU_MEMORY_GPU_H
#define TENSORSTACK_KERNELS_GPU_MEMORY_GPU_H

#include "global/hard_allocator.h"
#include "global/hard_converter.h"

namespace ts {
    void *gpu_allocator(int id, size_t new_size, void *mem, size_t mem_size);

    void gpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size);
    void cpu2gpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size);
    void gpu2cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size);

    void gpu2gpu(void *dst, const void *src, size_t size);
    void cpu2gpu(void *dst, const void *src, size_t size);
    void gpu2cpu(void *dst, const void *src, size_t size);
}


#endif //TENSORSTACK_KERNELS_GPU_MEMORY_GPU_H
