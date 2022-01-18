//
// Created by kier on 2019/4/8.
//

#ifndef TENSORSTACK_KERNELS_CPU_STRIDED_SLICE_H
#define TENSORSTACK_KERNELS_CPU_STRIDED_SLICE_H

#include "operator_on_cpu.h"
#include "backend/base/base_strided_slice.h"


namespace ts {
    namespace cpu {
        class StridedSlice : public OperatorOnCPU<base::StridedSlice> {
        public:
            using self = StridedSlice;
            using supper = OperatorOnCPU<base::StridedSlice>;

            void strided_slice(
                    const Tensor &x,
                    const std::vector<int> &begin,
                    const std::vector<int> &end,
                    const std::vector<int> &stride,
                    Tensor &out) override;
        };
    }
}

#endif  // TENSORSTACK_KERNELS_CPU_STRIDED_SLICE_H
