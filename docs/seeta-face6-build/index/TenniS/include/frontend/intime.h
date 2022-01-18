//
// Created by kier on 2019-04-13.
//

#ifndef TENSORSTACK_FRONTEND_INTIME_H
#define TENSORSTACK_FRONTEND_INTIME_H

#include "core/tensor.h"
#include "module/bubble.h"
#include "desc.h"

#include <string>
#include <vector>
#include <array>

#include <module/graph.h>
#include <runtime/stack.h>
#include <runtime/workbench.h>


namespace ts {
    namespace intime {
        /**
         * @note: the returned tensor is using borrowed memory from bench, use clone or free, before bench finalization
         */
        TS_DEBUG_API Tensor run(Workbench &bench, const Bubble &bubble, const std::vector<Tensor> &inputs);

        /**
         * @note: the returned tensor is using borrowed memory from bench, use clone or free, before bench finalization
         */
        TS_DEBUG_API Tensor run(const Bubble &bubble, const std::vector<Tensor> &inputs);

        TS_DEBUG_API Tensor resize2d(const Tensor &x, const Tensor &size,
                                     desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor
        resize2d(const Tensor &x, const std::vector<int32_t> &size, desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor add(const Tensor &lhs, const Tensor &rhs);

        TS_DEBUG_API Tensor sub(const Tensor &lhs, const Tensor &rhs);

        TS_DEBUG_API Tensor mul(const Tensor &lhs, const Tensor &rhs);

        TS_DEBUG_API Tensor div(const Tensor &lhs, const Tensor &rhs);

        TS_DEBUG_API Tensor transpose(const Tensor &x, const std::vector<int32_t> &permute);

        TS_DEBUG_API Tensor sigmoid(const Tensor &x);

        TS_DEBUG_API Tensor gather(const Tensor &x, const Tensor &indices, int32_t axis);

        TS_DEBUG_API Tensor gather(const Tensor &x, const std::vector<int32_t> &indices, int32_t axis);

        TS_DEBUG_API Tensor concat(const std::vector<Tensor> &x, int32_t dim);

        TS_DEBUG_API Tensor softmax(const Tensor &x, int32_t dim, bool smooth = true);

        TS_DEBUG_API Tensor pad(const Tensor &x, const Tensor &padding, float padding_value = 0);

        TS_DEBUG_API Tensor cast(const Tensor &x, DTYPE dtype);

        TS_DEBUG_API Tensor affine_sample2d(const Tensor &x, const Tensor &size, const Tensor &affine,
                                            int32_t dim = -1,
                                            float outer_value = 0,
                                            desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_sample2d(const std::string &name,
                                            const Tensor &x,
                                            const std::array<int32_t, 2> &size, const Tensor &affine,
                                            int32_t dim = -1,
                                            float outer_value = 0,
                                            desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_sample2d(const std::string &name,
                                            const Tensor &x,
                                            const Tensor &size, const std::array<float, 9> &affine,
                                            int32_t dim = -1,
                                            float outer_value = 0,
                                            desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_sample2d(const std::string &name,
                                            const Tensor &x,
                                            const std::array<int32_t, 2> &size, const std::array<float, 9> &affine,
                                            int32_t dim = -1,
                                            float outer_value = 0,
                                            desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_on_sample2d(const Tensor &x, const Tensor &size, const Tensor &affine,
                                               int32_t dim = -1,
                                               desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_on_sample2d(const std::string &name,
                                               const Tensor &x,
                                               const std::array<int32_t, 2> &size, const Tensor &affine,
                                               int32_t dim = -1,
                                               desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_on_sample2d(const std::string &name,
                                               const Tensor &x,
                                               const Tensor &size, const std::array<float, 9> &affine,
                                               int32_t dim = -1,
                                               desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Tensor affine_on_sample2d(const std::string &name,
                                               const Tensor &x,
                                               const std::array<int32_t, 2> &size, const std::array<float, 9> &affine,
                                               int32_t dim = -1,
                                               desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API int64_t memcpy(
                Tensor &dst_desc, void *dst_data, int64_t dst_shift,
                const Tensor &src_desc, const void *src_data, int64_t src_shift,
                int64_t size);

        TS_DEBUG_API Tensor matmul(const Tensor &A, const Tensor &B, bool transpose = false);

        TS_DEBUG_API Tensor broadcast(const Tensor &x, const Tensor &shape);

        TS_DEBUG_API Tensor broadcast(const Tensor &x, const std::vector<int32_t> &shape);
    }
}

#endif //TENSORSTACK_FRONTEND_INTIME_H
