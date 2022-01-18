//
// Created by yang on 2019/9/11.
//

#ifndef TENSORSTACK_KERNELS_CPU_POOLING_ALGORITHM_H
#define TENSORSTACK_KERNELS_CPU_POOLING_ALGORITHM_H

#include <core/tensor.h>
#include <backend/common_structure.h>

namespace ts {
    namespace cpu {
        //TODO:support other case
        template<typename T>
        class TS_DEBUG_API PoolingAlgorithm{
        public:
            //NOTE: onlu support ksize=3,stride=2,pad=1or0,NCHW
            static void max_pooling_k3s2(const Tensor &input,
                                         Tensor &out,
                                         const Padding2D &padding);

            static void max_pooling_k2s2(const Tensor &input,
                                         Tensor &out,
                                         const Padding2D &padding);
        };
    }//cpu
}//ts

#endif //TENSORSTACK_KERNELS_CPU_POOLING_ALGORITHM_H
