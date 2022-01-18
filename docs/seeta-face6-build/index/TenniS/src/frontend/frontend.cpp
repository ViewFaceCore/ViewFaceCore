//
// Created by kier on 2019/3/27.
//

#include "frontend/frontend.h"

#include "module/menu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace frontend {
        NodeOrTensor::NodeOrTensor(const Node &node)
                : m_node(node) {}

        NodeOrTensor::NodeOrTensor(const Tensor &tensor)
                : m_node(bubble::data("", tensor)) {}

        NodeOrTensor::NodeOrTensor(const Tensor &tensor, const char *device)
                : m_node(bubble::data("", tensor, device)) {}

        NodeOrTensor::NodeOrTensor(const Tensor &tensor, const DeviceType &device)
                : m_node(bubble::data("", tensor, device)) {}

        NodeOrTensor::NodeOrTensor(const Tensor &tensor, const MemoryDevice &device)
                : m_node(bubble::data("", tensor, device.type())) {}

        NodeOrTensor::operator Node() const {
            return m_node;
        }

        Node pad(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &padding,
                 float padding_value) {
            return symbol::pad(name, x, padding, padding_value);
        }

        Node pad(const std::string &name, const NodeOrTensor &x, const std::vector<DimPadding> &padding,
                 float padding_value) {
            TS_AUTO_CHECK(!padding.empty());
            auto padding_tensor = tensor::build(INT32, {int(padding.size()), 2}, &padding[0].first);
            return pad(name, x, {padding_tensor, CPU}, padding_value);
        }

        Node resize2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size, desc::ResizeType type) {
            return symbol::resize2d(name, x, size, type);
        }

        Node resize2d(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &size,
                      desc::ResizeType type) {
            TS_AUTO_CHECK(!size.empty());
            auto size_tensor = tensor::build(INT32, size.size(), size.data());
            return resize2d(name, x, {size_tensor, CPU}, type);
        }

        Node add(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs) {
            return symbol::add(name, lhs, rhs);
        }

        Node sub(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs) {
            return symbol::sub(name, lhs, rhs);
        }

        Node mul(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs) {
            return symbol::mul(name, lhs, rhs);
        }

        Node div(const std::string &name, const NodeOrTensor &lhs, const NodeOrTensor &rhs) {
            return symbol::div(name, lhs, rhs);
        }

        Node transpose(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &permute) {
            return symbol::transpose(name, x, permute);
        }

        Node sigmoid(const std::string &name, const NodeOrTensor &x) {
            return symbol::sigmoid(name, x);
        }

        Node gather(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &indices, int32_t axis) {
            return symbol::gather(name, x, indices, axis);
        }

        Node concat(const std::string &name, const std::vector<NodeOrTensor> &x, int32_t dim) {
            std::vector<Node> nodex(x.begin(), x.end());
            return symbol::concat(name, nodex, dim);
        }

        Node softmax(const std::string &name, const NodeOrTensor &x, int32_t dim, bool smooth) {
            return symbol::softmax(name, x, dim, smooth);
        }

        Node cast(const std::string &name, const NodeOrTensor &x, DTYPE dtype) {
            return symbol::cast(name, x, dtype);
        }

        Node affine_sample2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size,
                             const NodeOrTensor &affine, int32_t dim, float outer_value, desc::ResizeType type) {
            return symbol::affine_sample2d(name, x, size, affine, dim, outer_value, type);
        }

        Node affine_sample2d(const std::string &name, const NodeOrTensor &x, const std::array<int32_t, 2> &size,
                             const NodeOrTensor &affine, int32_t dim, float outer_value, desc::ResizeType type) {
            return affine_sample2d(name, x,
                                   tensor::build(INT32, Shape({2,}), &size[0]),
                                   affine,
                                   dim, outer_value, type);
        }

        Node affine_sample2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size,
                             const std::array<float, 9> &affine, int32_t dim, float outer_value,
                             desc::ResizeType type) {
            return affine_sample2d(name, x,
                                   size,
                                   tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                   dim, outer_value, type);
        }

        Node affine_sample2d(const std::string &name, const NodeOrTensor &x, const std::array<int32_t, 2> &size,
                             const std::array<float, 9> &affine, int32_t dim, float outer_value,
                             desc::ResizeType type) {
            return affine_sample2d(name, x,
                                   tensor::build(INT32, Shape({2,}), &size[0]),
                                   tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                   dim, outer_value, type);
        }

        Node affine_on_sample2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size,
                                const NodeOrTensor &affine, int32_t dim, desc::ResizeType type) {
            return symbol::affine_on_sample2d(name, x, size, affine, dim, type);
        }

        Node affine_on_sample2d(const std::string &name, const NodeOrTensor &x, const std::array<int32_t, 2> &size,
                                const NodeOrTensor &affine, int32_t dim, desc::ResizeType type) {
            return affine_on_sample2d(name, x,
                                      tensor::build(INT32, Shape({2,}), &size[0]),
                                      affine,
                                      dim, type);
        }

        Node affine_on_sample2d(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &size,
                                const std::array<float, 9> &affine, int32_t dim,
                                desc::ResizeType type) {
            return affine_on_sample2d(name, x,
                                      size,
                                      tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                      dim, type);
        }

        Node affine_on_sample2d(const std::string &name, const NodeOrTensor &x, const std::array<int32_t, 2> &size,
                                const std::array<float, 9> &affine, int32_t dim,
                                desc::ResizeType type) {
            return affine_on_sample2d(name, x,
                                      tensor::build(INT32, Shape({2,}), &size[0]),
                                      tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                      dim, type);
        }

        Node matmul(const std::string &name, const NodeOrTensor &A, const NodeOrTensor &B, bool transpose) {
            return symbol::matmul(name, A, B, transpose);
        }

        Node broadcast(const std::string &name, const NodeOrTensor &x, const NodeOrTensor &shape) {
            return symbol::broadcast(name, x, shape);
        }

        Node broadcast(const std::string &name, const NodeOrTensor &x, const std::vector<int32_t> &shape) {
            auto shape_tensor = tensor::build(INT32, shape.size(), shape.data());
            return broadcast(name, x, {shape_tensor, CPU});
        }
    }
}