//
// Created by kier on 2019-04-13.
//

#include <core/tensor_builder.h>
#include <api/intime.h>

#include "frontend/intime.h"
#include "runtime/operator.h"
#include "global/operator_factory.h"

#include "runtime/stack.h"

#include "utils/ctxmgr_lite.h"
#include "utils/need.h"

namespace ts {
    namespace intime {
        Tensor run(Workbench &bench, const Bubble &bubble, const std::vector<Tensor> &inputs) {
            auto &stack = bench.stack();
            stack.push_base(int(stack.size()));
            need pop_base(&Stack::pop_base, &stack);
            need clear_stack(&Stack::clear, &stack);

            bench.online_run(bubble, inputs);

            auto fields_count = stack.size();
            if (fields_count == 1) {
                return stack[0];
            }
            std::vector<Tensor> fields(fields_count);
            for (int i = 0; i < fields_count; ++i) {
                fields[i] = stack[i];
            }
            Tensor output;
            output.pack(fields);
            return std::move(output);
        }

        Tensor run(const Bubble &bubble, const std::vector<Tensor> &inputs) {
            auto bench = ctx::get<Workbench>();
            if (bench == nullptr) {
                TS_LOG_ERROR << "Must bind Workbench before run" << eject;
            }
            return run(*bench, bubble, inputs);
        }

        Tensor resize2d(const Tensor &x, const Tensor &size, desc::ResizeType type) {
            return run(desc::resize2d(type), {x, size});
        }

        Tensor resize2d(const Tensor &x, const std::vector<int32_t> &size, desc::ResizeType type) {
            return resize2d(x, tensor::from(size), type);
        }

        Tensor add(const Tensor &lhs, const Tensor &rhs) {
            return run(desc::add(), {lhs, rhs});
        }

        Tensor sub(const Tensor &lhs, const Tensor &rhs) {
            return run(desc::sub(), {lhs, rhs});
        }

        Tensor mul(const Tensor &lhs, const Tensor &rhs) {
            return run(desc::mul(), {lhs, rhs});
        }

        Tensor div(const Tensor &lhs, const Tensor &rhs) {
            return run(desc::div(), {lhs, rhs});
        }

        Tensor transpose(const Tensor &x, const std::vector<int32_t> &permute) {
            return run(desc::transpose(permute), {x});
        }

        Tensor sigmoid(const Tensor &x) {
            return run(desc::sigmoid(), {x});
        }

        Tensor gather(const Tensor &x, const Tensor &indices, int32_t axis) {
            return run(desc::gather(axis), {x, indices});
        }

        Tensor gather(const Tensor &x, const std::vector<int32_t> &indices, int32_t axis) {
            return run(desc::gather(axis), {x, tensor::build(INT32, indices)});
        }

        Tensor concat(const std::vector<Tensor> &x, int32_t dim) {
            if (x.size() == 1) return x[0];
            return run(desc::concat(dim), x);
        }

        Tensor softmax(const Tensor &x, int32_t dim, bool smooth) {
            return run(desc::softmax(dim, smooth), {x});
        }

        Tensor pad(const Tensor &x, const Tensor &padding, float padding_value) {
            return run(desc::pad(padding_value), {x, padding});
        }

        Tensor cast(const Tensor &x, DTYPE dtype) {
            if (x.dtype() == dtype) return x;
            return run(desc::cast(dtype), {x});
        }

        Tensor
        affine_sample2d(const Tensor &x, const Tensor &size, const Tensor &affine, int32_t dim, float outer_value,
                        desc::ResizeType type) {
            return run(desc::affine_sample2d(dim, outer_value, type), {x, size, affine});
        }

        Tensor affine_sample2d(const std::string &name, const Tensor &x, const std::array<int32_t, 2> &size,
                               const Tensor &affine, int32_t dim, float outer_value, desc::ResizeType type) {
            return affine_sample2d(x,
                                   tensor::build(INT32, Shape({2,}), &size[0]),
                                   affine,
                                   dim, outer_value, type);
        }

        Tensor affine_sample2d(const std::string &name, const Tensor &x, const Tensor &size,
                               const std::array<float, 9> &affine, int32_t dim, float outer_value,
                               desc::ResizeType type) {
            return affine_sample2d(x,
                                   size,
                                   tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                   dim, outer_value, type);
        }

        Tensor affine_sample2d(const std::string &name, const Tensor &x, const std::array<int32_t, 2> &size,
                               const std::array<float, 9> &affine, int32_t dim, float outer_value,
                               desc::ResizeType type) {
            return affine_sample2d(x,
                                   tensor::build(INT32, Shape({2,}), &size[0]),
                                   tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                   dim, outer_value, type);
        }

        Tensor
        affine_on_sample2d(const Tensor &x, const Tensor &size, const Tensor &affine, int32_t dim,
                           desc::ResizeType type) {
            return run(desc::affine_on_sample2d(dim, type), {x, size, affine});
        }

        Tensor affine_on_sample2d(const std::string &name, const Tensor &x, const std::array<int32_t, 2> &size,
                                  const Tensor &affine, int32_t dim, desc::ResizeType type) {
            return affine_on_sample2d(x,
                                      tensor::build(INT32, Shape({2,}), &size[0]),
                                      affine,
                                      dim, type);
        }

        Tensor affine_on_sample2d(const std::string &name, const Tensor &x, const Tensor &size,
                                  const std::array<float, 9> &affine, int32_t dim,
                                  desc::ResizeType type) {
            return affine_on_sample2d(x,
                                      size,
                                      tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                      dim, type);
        }

        Tensor affine_on_sample2d(const std::string &name, const Tensor &x, const std::array<int32_t, 2> &size,
                                  const std::array<float, 9> &affine, int32_t dim,
                                  desc::ResizeType type) {
            return affine_on_sample2d(x,
                                      tensor::build(INT32, Shape({2,}), &size[0]),
                                      tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                                      dim, type);
        }

        int64_t memcpy(
                Tensor &dst_desc, void *dst_data, int64_t dst_shift,
                const Tensor &src_desc, const void *src_data, int64_t src_shift,
                int64_t size) {
            if (dst_data == nullptr) dst_data = dst_desc.data();
            if (src_data == nullptr) src_data = src_desc.data();
            auto copied = memcpy(reinterpret_cast<char *>(dst_data) + dst_shift, dst_desc.device(), size_t(size),
                                 reinterpret_cast<const char *>(src_data) + src_shift, src_desc.device(), size_t(size));
            return int64_t(copied);
        }

        Tensor matmul(const Tensor &A, const Tensor &B, bool transpose) {
            return run(desc::matmul(transpose), {A, B});
        }

        Tensor broadcast(const Tensor &x, const Tensor &shape) {
            return run(desc::broadcast(), {x, shape});
        }

        Tensor broadcast(const Tensor &x, const std::vector<int32_t> &shape) {
            return run(desc::broadcast(), {x, tensor::from(shape)});
        }
    }
}
