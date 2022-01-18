//
// Created by kier on 2018/7/18.
//

#ifndef TENSORSTACK_KERNELS_CPU_MEMORY_CPU_H
#define TENSORSTACK_KERNELS_CPU_MEMORY_CPU_H

#include "global/hard_allocator.h"
#include "global/hard_converter.h"

namespace ts {
    void *cpu_allocator(int id, size_t new_size, void *mem, size_t mem_size);

    void cpu_converter(int dst_id, void *dst, int src_id, const void *src, size_t size);
}


#endif //TENSORSTACK_KERNELS_CPU_MEMORY_CPU_H
