//
// Created by kier on 2019/3/27.
//

#include "frontend/symbol.h"

#include "module/menu.h"
#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace symbol {
        Node pad(const std::string &name, const Node &x, const Node &padding, float padding_value) {
            Node node = bubble::op(name, name::layer::pad(), {x, padding});
            node->set(name::padding_value, tensor::from<float>(padding_value));
            return node;
        }

        Node resize2d(const std::string &name, const Node &x, const Node &size, desc::ResizeType type) {
            Node node = bubble::bubble(desc::resize2d(type), name);
            Node::Link(node, {x, size});
            return node;
        }

        Node add(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::add(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node sub(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::sub(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node mul(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::mul(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node div(const std::string &name, const Node &lhs, const Node &rhs) {
            Node node = bubble::bubble(desc::div(), name);
            Node::Link(node, {lhs, rhs});
            return node;
        }

        Node transpose(const std::string &name, const Node &x, const std::vector<int32_t> &permute) {
            Node node = bubble::bubble(desc::transpose(permute), name);
            Node::Link(node, {x});
            return node;
        }

        Node sigmoid(const std::string &name, const Node &x) {
            Node node = bubble::bubble(desc::sigmoid(), name);
            Node::Link(node, {x});
            return node;
        }

        Node gather(const std::string &name, const Node &x, const Node &indices, int32_t axis) {
            Node node = bubble::bubble(desc::gather(axis), name);
            Node::Link(node, {x, indices});
            return node;
        }

        Node concat(const std::string &name, const std::vector<Node> &x, int32_t dim) {
            Node node = bubble::bubble(desc::gather(dim), name);
            Node::Link(node, x);
            return node;
        }

        Node softmax(const std::string &name, const Node &x, int32_t dim, bool smooth) {
            Node node = bubble::bubble(desc::softmax(dim, smooth), name);
            Node::Link(node, {x});
            return node;
        }

        Node cast(const std::string &name, const Node &x, DTYPE dtype) {
            Node node = bubble::bubble(desc::cast(dtype), name);
            Node::Link(node, {x});
            return node;
        }

        Node affine_sample2d(const std::string &name, const Node &x, const Node &size, const Node &affine, int32_t dim,
                             float outer_value, desc::ResizeType type) {
            Node node = bubble::bubble(desc::affine_sample2d(dim, outer_value, type), name);
            Node::Link(node, {x, size, affine});
            return node;
        }

        Node affine_on_sample2d(const std::string &name, const Node &x, const Node &size, const Node &affine,
                                int32_t dim, desc::ResizeType type) {
            Node node = bubble::bubble(desc::affine_on_sample2d(dim, type), name);
            Node::Link(node, {x, size, affine});
            return node;
        }

        Node matmul(const std::string &name, const Node &A, const Node &B, bool transpose) {
            Node node = bubble::bubble(desc::matmul(transpose), name);
            Node::Link(node, {A, B});
            return node;
        }

        Node broadcast(const std::string &name, const Node &x, const Node &shape) {
            Node node = bubble::bubble(desc::matmul(), name);
            Node::Link(node, {x, shape});
            return node;
        }
    }
}
