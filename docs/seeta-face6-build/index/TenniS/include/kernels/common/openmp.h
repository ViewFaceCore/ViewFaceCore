//
// Created by kier on 2018/12/21.
//

#ifndef TENSORSTACK_KERNELS_COMMON_OPENMP_H
#define TENSORSTACK_KERNELS_COMMON_OPENMP_H

#ifdef TS_USE_OPENMP
#include <omp.h>
#include "runtime/runtime.h"
#include <algorithm>

// #define TS_OPENMP_BLOCK_SIZE 10240

#endif

namespace ts {
    inline int openmp_threads(const int = 0) {
#ifdef TS_USE_OPENMP
        auto max_threads = omp_get_num_procs();
        auto runtime = ctx::ptr<RuntimeContext>();
        if (runtime == nullptr) return max_threads;
        if (runtime->get_computing_thread_number() <= 0) return max_threads;
        return runtime->get_computing_thread_number();
#else
        return 1;
#endif
    }

    inline int openmp_thread_id() {
#ifdef TS_USE_OPENMP
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
}





#endif //TENSORSTACK_KERNELS_COMMON_OPENMP_H
