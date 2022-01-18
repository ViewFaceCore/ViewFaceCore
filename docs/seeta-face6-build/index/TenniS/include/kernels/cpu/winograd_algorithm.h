//
// Created by yang on 2019/10/21.
//

#ifndef TENSORSTACK_KERNELS_CPU_WINOGRAD_ALGORITHM_H
#define TENSORSTACK_KERNELS_CPU_WINOGRAD_ALGORITHM_H

#include "core/tensor.h"

namespace  ts{
    namespace cpu{
        template <typename T>
        class TS_DEBUG_API Conv2dWinogradAlgorithm{
        public:
            static void winograd_f23_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm);

            static void winograd_f23_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm);

            static void winograd_f23_transform_output(const Tensor& out_tm, int tile_count, Tensor& out);

            static void winograd_f23(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &kernel,
                                     Tensor &out,
                                     bool kernel_transformed = true);

            static void winograd_f63_transform_and_pack_kernel(const Tensor& kernel, int in_tile_size, Tensor &kernel_tm);

            static void winograd_f63_transform_and_pack_input(const Tensor& x, int tile_count, Tensor &x_tm);

            static void winograd_f63_transform_output(const Tensor& out_tm, int tile_count, Tensor& out);

            static void winograd_f63(const Tensor &x,
                                     const Padding2D &padding,
                                     float padding_value,
                                     const Tensor &kernel,
                                     Tensor &out,
                                     bool kernel_transformed = true);
        };
    }
}

#endif //TENSORSTACK_KERNELS_CPU_WINOGRAD_ALGORITHM_H
