//
// Created by kier on 2019/3/27.
//

#ifndef TENSORSTACK_FRONTEND_FRONTEND_H
#define TENSORSTACK_FRONTEND_FRONTEND_H

#include "module/menu.h"

#include "symbol.h"

#include <module/graph.h>

#include "frontend/desc.h"
#include <array>

namespace ts {
    namespace frontend {
        using namespace bubble;

        struct DimPadding {
            DimPadding() = default;

            DimPadding(int32_t first, int32_t second)
                    : first(first), second(second) {}

            int32_t first = 0;
            int32_t second = 0;
        };

        class TS_DEBUG_API NodeOrTensor {
        public:
            using self = NodeOrTensor;

            NodeOrTensor(const Node &node);

            NodeOrTensor(const Tensor &tensor);

            NodeOrTensor(const Tensor &tensor, const char *device);

            NodeOrTensor(const Tensor &tensor, const DeviceType &device);

            NodeOrTensor(const Tensor &tensor, const MemoryDevice &device);

            operator Node() const;

        private:
            Node m_node;
        };

        TS_DEBUG_API Node
        pad(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &padding, float padding_value = 0);

        TS_DEBUG_API Node pad(const std::string &name, const NodeOrTensor &x, const std::vector<DimPadding> &padding,
                              float padding_value = 0);

        TS_DEBUG_API Node resize2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size,
                                   desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node resize2d(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &size,
                                   desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node add(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs);

        TS_DEBUG_API Node sub(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs);

        TS_DEBUG_API Node mul(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs);

        TS_DEBUG_API Node div(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs);

        TS_DEBUG_API Node
        transpose(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &permute);

        TS_DEBUG_API Node sigmoid(const std::string &name, const NodeOrTensor &x);

        TS_DEBUG_API Node
        gather(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &indices, int32_t axis);

        TS_DEBUG_API Node concat(const std::string &name, const std::vector<NodeOrTensor> &x, int32_t dim);

        TS_DEBUG_API Node softmax(const std::string &name, const NodeOrTensor &x, int32_t dim, bool smooth = true);

        TS_DEBUG_API Node cast(const std::string &name, const NodeOrTensor &x, DTYPE dtype);

        TS_DEBUG_API Node affine_sample2d(const std::string &name,
                                          const NodeOrTensor &x, const NodeOrTensor &size, const NodeOrTensor &affine,
                                          int32_t dim = -1,
                                          float outer_value = 0,
                                          desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_sample2d(const std::string &name,
                                          const NodeOrTensor &x,
                                          const std::array<int32_t, 2> &size, const NodeOrTensor &affine,
                                          int32_t dim = -1,
                                          float outer_value = 0,
                                          desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_sample2d(const std::string &name,
                                          const NodeOrTensor &x,
                                          const NodeOrTensor &size, const std::array<float, 9> &affine,
                                          int32_t dim = -1,
                                          float outer_value = 0,
                                          desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_sample2d(const std::string &name,
                                          const NodeOrTensor &x,
                                          const std::array<int32_t, 2> &size, const std::array<float, 9> &affine,
                                          int32_t dim = -1,
                                          float outer_value = 0,
                                          desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_on_sample2d(const std::string &name,
                                             const NodeOrTensor &x, const NodeOrTensor &size,
                                             const NodeOrTensor &affine,
                                             int32_t dim = -1,
                                             desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_on_sample2d(const std::string &name,
                                             const NodeOrTensor &x,
                                             const std::array<int32_t, 2> &size, const NodeOrTensor &affine,
                                             int32_t dim = -1,
                                             desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_on_sample2d(const std::string &name,
                                             const NodeOrTensor &x,
                                             const NodeOrTensor &size, const std::array<float, 9> &affine,
                                             int32_t dim = -1,
                                             desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node affine_on_sample2d(const std::string &name,
                                             const NodeOrTensor &x,
                                             const std::array<int32_t, 2> &size, const std::array<float, 9> &affine,
                                             int32_t dim = -1,
                                             desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Node matmul(const std::string &name, const NodeOrTensor &A, const NodeOrTensor &B, bool transpose = false);

        TS_DEBUG_API Node broadcast(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &shape);

        TS_DEBUG_API Node broadcast(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &shape);
    }
}


#endif //TENSORSTACK_FRONTEND_FRONTEND_H
