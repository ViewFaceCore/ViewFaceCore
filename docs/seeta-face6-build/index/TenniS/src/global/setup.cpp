//
// Created by lby on 2018/3/12.
//

#include "global/setup.h"
#include "kernels/cpu/memory_cpu.h"

#ifdef TS_USE_CUDA
#include "kernels/gpu/memory_gpu.h"
#endif

namespace ts {
    void setup() {
        // may do some setup
        cpu_allocator(0, 0, nullptr, 0);
#ifdef TS_USE_CUDA
        gpu_allocator(0, 0, nullptr, 0);
#endif
    }
}

