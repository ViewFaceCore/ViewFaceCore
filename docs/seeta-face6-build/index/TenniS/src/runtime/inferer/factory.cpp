//
// Created by kier on 2019/11/22.
//

#include <core/tensor_builder.h>
#include "global/shape_inferer_factory.h"
#include "utils/static.h"
#include "backend/name.h"

#include "runtime/inferer.h"

#include "utils/box.h"
#include <algorithm>
#include <global/operator_factory.h>

#include "runtime/stack.h"
#include <numeric>

namespace ts {
    namespace infer_factory {
        static Tensor get_value(const Node &node) {
            if (node->op() == Bubble::Const) {
                return node->get("value");
            }
            if (node->has("#value")) {
                return node->get("#value");
            }
            return Tensor();
        }

        static Tensor try_run(const Node &node, const std::vector<Tensor> &inputs) {
            if (node->op() == Bubble::Const) {
                return node->get("value");
            }

            if (Bubble::IsEndPoint(node->op())) {
                return Tensor();
            }

            for (auto &input : inputs) {
                if (input.empty()) return Tensor();
            }

            MemoryDevice memory_device(CPU);
            Stack stack(memory_device);

            auto op = OperatorCreator::CreateNoException(memory_device.type(), node->op());
            if (op == nullptr) return Tensor();

            Tensor output;

            try {
                for (const auto &it : node->params()) {
                    op->set(it.first, it.second);
                }
                op->init();
                for (auto &t : inputs) {
                    stack.push(t);
                }
                auto out = op->run(stack);
                stack.erase(0, -out);
                if (out == 1) {
                    output = stack[0];
                } else {
                    output = Tensor::Pack(std::vector<Tensor>(stack.begin(), stack.end()));
                }
            } catch (...) {
                return Tensor();
            }

            return output;
        }

        static TensorPrototype _param(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (!node->has("#shape")) {
                throw Exception(node->op() + ":" + node->name() + " must set #shape");
                // return TensorPrototype();
            }
            auto dtype = FLOAT32;
            if (node->has("#dtype")) {
                dtype = DTYPE(tensor::to_int(node->get("#dtype")));
            }
            auto shape = tensor::array::to_int(node->get("#shape"));
            return TensorPrototype(dtype, shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "<param>", _param)

        static TensorPrototype _const(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto value = node->get("value");
            return value.proto();
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "<const>", _const)

        static TensorPrototype _resize2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return VOID;
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_resize2d", _resize2d)

        static TensorPrototype _transpose(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            std::vector<int32_t> permute;
            if (node->has("permute")) {
                permute = node->get_int_list("permute");
            } else {
                if (x.dims() < 2) return x;
                std::vector<int32_t> tmp;
                for (size_t i = 0; i < x.dims(); ++i) {
                    tmp.push_back(int32_t(i));
                }
                std::swap(tmp[tmp.size() - 1], tmp[tmp.size() - 2]);
            }

            auto x_shape = x.sizes();
            while (x_shape.size() < permute.size()) {
                x_shape.insert(x_shape.begin(), 1);
            }
            auto y_shape = std::vector<int32_t>(permute.size());

            for (size_t i = 0; i < permute.size(); ++i) {
                if (permute[i] >= x_shape.size()) return VOID;
                y_shape[i] = x_shape[permute[i]];
            }

            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_transpose", _transpose)

        static TensorPrototype _copy(const Node &node, const std::vector<TensorPrototype> &inputs) {
            return inputs[0];
        }

        static TensorPrototype to_float(const Node &node, const std::vector<TensorPrototype> &inputs) {
            return {FLOAT32, inputs[0].sizes()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "to_float", to_float)

        static TensorPrototype crop_nd(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return VOID;
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "crop_nd", crop_nd)

        static int32_t conv2d_forward(int32_t x, int32_t padding, int32_t dilation, int32_t kernel, int32_t stride) {
            return int32_t(std::floor((x + padding - (dilation * (kernel - 1) + 1)) / stride + 1));
        }

        static TensorPrototype conv2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d", conv2d)

        TS_STATIC_ACTION(ShapeInferer::Register, "add_bias", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "relu", _copy)

        static int32_t pooling2d_forward(int32_t x, int32_t padding, int32_t kernel, int32_t stride) {
            return int32_t(std::ceil((x + padding - kernel) / float(stride) + 1));
        }

        static TensorPrototype pooling2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto padding = node->get_int_list("padding");
            auto ksize = node->get_int_list("ksize");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = pooling2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        ksize[dim],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "pooling2d", pooling2d)

        static int32_t eltwise_infer_dim(int32_t a, int32_t b) {

            if (a <= 0) {
                if (b == 1)
                    return -1;
                else
                    return b;
            }
            if (a == 1)
                return b;
            if (b <= 0)
                return a;
            if (b == 1)
                return a;
            if (a == b)
                return a;
            return -1;
        }

        static void begin_insert_ones(Shape &x, size_t n) {
            auto ones = Shape(n, 1);
            x.insert(x.begin(), ones.begin(), ones.end());
        }

        static TensorPrototype _eltwise(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto dtype = inputs[0].dtype();

            auto lhs_shape = inputs[0].sizes();
            auto rhs_shape = inputs[1].sizes();

            if (lhs_shape.size() > rhs_shape.size()) {
                begin_insert_ones(rhs_shape, lhs_shape.size() - rhs_shape.size());
            } else if (rhs_shape.size() > lhs_shape.size()) {
                begin_insert_ones(lhs_shape, rhs_shape.size() - lhs_shape.size());
            }

            auto dims = lhs_shape.size();
            auto out_shape = std::vector<int32_t>(dims, -1);

            for (size_t i = 0; i < dims; ++i) {
                out_shape[i] = eltwise_infer_dim(lhs_shape[i], rhs_shape[i]);
            }

            return {dtype, out_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "add", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "sub", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "mul", _eltwise)
        TS_STATIC_ACTION(ShapeInferer::Register, "div", _eltwise)

        static TensorPrototype flatten(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto dim = node->has("dim") ? node->get_int("dim") : 1;
            if (dim < 0) dim += int32_t(x.dims());

            std::vector<int> y_shape;

            auto need_size = size_t(dim + 1);
            auto x_size = x.sizes().size();
            if (need_size < x_size) {
                auto &size = x.sizes();
                std::vector<int> shape(size.begin(), size.begin() + need_size);
                shape.back() = std::accumulate(size.begin() + dim, size.end(), 1, std::multiplies<int>());
                y_shape = std::move(shape);
            } else if (need_size > x_size) {
                std::vector<int> ones(need_size - x_size, 1);
                auto shape = x.sizes();
                shape.insert(shape.end(), ones.begin(), ones.end());
                y_shape = shape.std();
            } else {
                y_shape = x.sizes().std();
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "flatten", flatten)

        static TensorPrototype flatten2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto dim = node->has("dim") ? node->get_int("dim") : 1;
            if (dim < 0) dim += int32_t(x.dims());

            auto &size = x.sizes();
            std::vector<int> y_shape;

            if (dim <= 0) {
                y_shape = {1, std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>())};
            } else if (dim >= x.dims()) {
                y_shape = {std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>()), 1};
            } else {
                y_shape = {std::accumulate(size.begin(), size.begin() + dim, 1, std::multiplies<int>()),
                           std::accumulate(size.begin() +  dim, size.end(), 1, std::multiplies<int>())};
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "flatten2d", flatten2d)

        static TensorPrototype inner_prod(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            auto &w = inputs[1];

            bool transpose = false;
            if (node->has("transpose")) {
                transpose = node->get_bool("transpose");
            }

            if (transpose) {
                return {x.dtype(), {x.size(0), w.size(0)}};
            } else {
                return {x.dtype(), {x.size(0), w.size(1)}};
            }
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "inner_prod", inner_prod)

        static bool valid_dims(const Shape &dims) {
            for (auto dim : dims) {
                if (dim <= 0) return false;
            }
            return true;
        }

        static TensorPrototype _reshape(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            Shape shape = node->get_int_list("shape");

            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == 0) {
                    if (i >= x.dims()) {
                        return VOID;
                    }
                    shape[i] = x.size(i);
                }
            }

            if (valid_dims(x.sizes())) {
                Tensor tmp(MemoryDevice("_fake_"), INT8, x.sizes());
                tmp = tmp.reshape(shape);
                shape = tmp.sizes();
            }

            return {x.dtype(), shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_reshape", _reshape)

        TS_STATIC_ACTION(ShapeInferer::Register, "softmax", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "batch_norm", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "batch_scale", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "fused_batch_norm", _copy)

        static TensorPrototype _cast(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto dtype = node->get_int("dtype");
            return {DTYPE(dtype), inputs[0].sizes()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_cast", _cast)

        static TensorPrototype dynamic_padding(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (node->name() == "_op_${EAGER}/${JOIN}/Tensor_111_dragon_padding") {
                TS_LOG_DEBUG << "Hook";
            }

            std::string format = "NCHW";
            if (node->has("format")) format = node->get_string("format");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dim;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dim = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dim = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> param0_shape(4, 1);
            for (auto dim: kernel_dims) {
                if (x.size(size_t(dim)) < 0) return VOID;
                param0_shape[dim] = x.size(size_t(dim));
            }

            if (param0_shape[channel_dim] < 0) {
                return VOID;
            }
            param0_shape[channel_dim] = x.size(size_t(channel_dim));

            std::vector<Tensor> run_inputs = {Tensor(FLOAT32, param0_shape)};
            for (size_t i = 1; i < inputs.size(); ++i) {
                run_inputs.push_back(get_value(node.input(i)));
            }

            auto output = try_run(node, run_inputs);
            if (!output.empty()) {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#value", output);
            }

            return {INT32, {4, 2}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_onnx_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_dragon_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_mx_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_tf_conv2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_tf_pooling2d_padding", dynamic_padding)
        TS_STATIC_ACTION(ShapeInferer::Register, "_dragon_conv2d_padding", dynamic_padding)

        static TensorPrototype pooling2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            if (padding_value.empty()) return VOID;
            auto ksize_value = get_value(node.input(2));
            if (ksize_value.empty()) return VOID;
            auto stride_value = get_value(node.input(3));
            if (stride_value.empty()) return VOID;

            auto padding = tensor::array::to_int(padding_value);
            auto stide = tensor::array::to_int(stride_value);
            auto ksize = tensor::array::to_int(ksize_value);

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = pooling2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        ksize[dim],
                        stide[dim]);
            }

            {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#padding", tensor::build<int32_t>(INT32, {4, 2}, padding));
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "pooling2d_v2", pooling2d_v2)

        static TensorPrototype conv2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            if (padding_value.empty()) return VOID;
            auto padding = tensor::array::to_int(padding_value);

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[2];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#padding", tensor::build<int32_t>(INT32, {4, 2}, padding));
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d_v2", conv2d_v2)

        static TensorPrototype gemm(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto A = inputs[0];
            auto B = inputs[1];


            auto transA = node->get_bool("transA");
            auto transB = node->get_bool("transB");

            auto M = transA ? A.size(1) : A.size(0);
            auto N = transB ? B.size(0) : B.size(1);

            return {A.dtype(), {M, N}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "gemm", gemm)

        static TensorPrototype concat(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto dim = node->get_int("dim");

            auto dtype = inputs[0].dtype();
            auto shape = inputs[0].sizes();

            if (dim < 0) dim = int32_t(shape.size()) + dim;
            if (dim < 0 || dim >= int32_t(shape.size())) return VOID;

            for (size_t i = 1; i < inputs.size(); ++i) {
                auto inc = inputs[i].size(dim);
                if (inc < 0) {
                    shape[dim] = -1;
                    break;
                }
                shape[dim] += inc;
            }

            return {dtype, shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "concat", concat)

        static TensorPrototype global_pooling2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = x.size(channel_dims);

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                y_shape[dim] = 1;
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "global_pooling2d", global_pooling2d)

        TS_STATIC_ACTION(ShapeInferer::Register, "sigmoid", _copy)

        static TensorPrototype _dims(const Node &node, const std::vector<TensorPrototype> &inputs) {
            {
                auto &update_node = *const_cast<Node*>(&node);
                auto dims = inputs[0].dims();
                update_node->set("#value", tensor::build<int32_t>(INT32, int32_t(dims)));
            }

            return {INT32, Shape()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_dims", _dims)

        static TensorPrototype _expand(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];

            auto dims_value = get_value(node.input(1));
            if (dims_value.empty()) return VOID;

            auto dims = size_t(tensor::to_int(dims_value));

            auto front = int32_t(dims);
            auto end = int32_t(dims);
            bool inverse = false;
            if (node->has("front")) front = node->get_int("front");
            if (node->has("end")) end = node->get_int("end");
            if (node->has("inverse")) inverse = node->get_bool("inverse");

            auto y = x.sizes();

            if (!inverse) {
                while (y.size() < dims && front > 0) {
                    y.insert(y.begin(), 1);
                }
                while (y.size() < dims && end > 0) {
                    y.insert(y.end(), 1);
                }
            } else {
                while (y.size() < dims && end > 0) {
                    y.insert(y.end(), 1);
                }
                while (y.size() < dims && front > 0) {
                    y.insert(y.begin(), 1);
                }
            }

            return {x.dtype(), y};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_expand", _expand)

        TS_STATIC_ACTION(ShapeInferer::Register, "_copy", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "abs", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "exp", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "l2_norm", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "norm_image", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "prelu", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "prewhiten", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "relu_max", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "rsqrt", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "sqrt", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "square", _copy)
        TS_STATIC_ACTION(ShapeInferer::Register, "tanh", _copy)

        static TensorPrototype _dimshuffle(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];

            auto dim = node->get_int("dim");
            auto shuffle = node->get_int_list("shuffle");

            if (dim < 0) dim += int32_t(x.dims());
            if (dim < 0 || dim >= x.dims()) return VOID;

            auto y_shape = x.sizes();
            y_shape[dim] = int32_t(shuffle.size());

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_dimshuffle", _dimshuffle)

        static TensorPrototype _limit(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size = node->get_int_list("shape");
            auto y_shape = x.sizes();
            if (size.size() != y_shape.size()) return VOID;
            for (size_t i = 0; i < size.size(); ++i) {
                if (size[i] > 0 && y_shape[i] > size[i]) y_shape[i] = size[i];
            }
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_limit", _limit)

        static TensorPrototype _nhwc_center_crop2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size = node->get_int_list("size");

            if (size.size() != 2 || x.dims() != 4) return VOID;

            auto W = size[0];
            auto H = size[1];

            auto y_shape = x.sizes();

            y_shape[1] = H;
            y_shape[2] = W;
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_nhwc_center_crop2d", _nhwc_center_crop2d)

        static TensorPrototype _nhwc_letterbox(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size = node->get_int_list("size");

            if (size.empty() || x.dims() != 4) return VOID;

            auto W = size[0];
            auto H = size.size() > 1 ? size[1] : W;

            auto y_shape = x.sizes();

            y_shape[1] = H;
            y_shape[2] = W;
            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_nhwc_letterbox", _nhwc_letterbox)

        static TensorPrototype _nhwc_scale_resize2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto x = inputs[0];
            auto size = node->get_int_list("size");

            if (size.empty() || x.dims() != 4) return VOID;

            auto y_shape = x.sizes();

            if (size.size() == 1) {
                auto S = size[0];
                auto H = y_shape[1];
                auto W = y_shape[2];

                if (H > W) {
                    y_shape[1] = S * H / W;
                    y_shape[2] = S;
                } else {
                    y_shape[1] = S;
                    y_shape[2] = S * W / H;
                }
            } else {
                auto W = size[0];
                auto H = size[1];

                y_shape[1] = H;
                y_shape[2] = W;
            }

            return TensorPrototype(x.dtype(), y_shape);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_nhwc_scale_resize2d", _nhwc_scale_resize2d)

        static TensorPrototype _reshape_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto shape_value = get_value(node.input(1));
            if (shape_value.empty()) return VOID;
            Shape shape = tensor::array::to_int(shape_value);

            for (size_t i = 0; i < shape.size(); ++i) {
                if (shape[i] == 0) {
                    if (i >= x.dims()) {
                        return VOID;
                    }
                    shape[i] = x.size(i);
                }
            }

            if (valid_dims(x.sizes())) {
                Tensor tmp(MemoryDevice("_fake_"), INT8, x.sizes());
                tmp = tmp.reshape(shape);
                shape = tmp.sizes();
            }

            return {x.dtype(), shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_reshape_v2", _reshape_v2)

        static TensorPrototype _shape(const Node &node, const std::vector<TensorPrototype> &inputs) {
            {
                auto &update_node = *const_cast<Node*>(&node);
                auto shape = inputs[0].sizes();
                update_node->set("#value", tensor::build<int32_t>(INT32, shape));
            }
            return {INT32, {int32_t(inputs[0].dims())}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_shape", _shape)

        static TensorPrototype affine_sample2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto size_value = get_value(node.input(1));
            if (size_value.empty()) return VOID;
            auto size = tensor::array::to_int(size_value);

            auto dim = -2;
            if (node->has("dim")) dim = node->get_int("dim");

            if (dim < 0) dim += int32_t(x.dims());
            if (dim < 0 || dim + 1 >= int32_t(x.dims())) return VOID;

            auto y_shape = x.sizes();

            y_shape[dim] = size[0];
            y_shape[dim + 1] = size[1];

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "affine_sample2d", affine_sample2d)

        static TensorPrototype argmax(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto dim = node->get_int("dim");

            if (dim < 0) dim += int32_t(x.dims());
            if (dim < 0 || dim>= int32_t(x.dims())) return VOID;

            auto y_shape = x.sizes();

            y_shape.erase(y_shape.begin() + dim);

            return {INT32, y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "argmax", argmax)

        static TensorPrototype batch_to_space4d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto crop = node->get_int_list("crop");
            auto block_shape = node->get_int_list("block_shape");

            if (crop.size() < 4 || block_shape.size() < 2) return VOID;

            auto block_height = block_shape[0];
            auto block_width = block_shape[1];
            auto crop_top = crop[0];
            auto crop_bottom = crop[1];
            auto crop_left = crop[2];
            auto crop_right = crop[3];

            auto &input_shape = x.sizes();
            std::vector<int32_t> output_shape(4, -1);

            output_shape[0] = input_shape[0] < 0 ? -1 : input_shape[0] / (block_height * block_width);
            output_shape[2] = input_shape[2] < 0 ? -1 : input_shape[2] * block_height - crop_top - crop_bottom;
            output_shape[3] = input_shape[3] < 0 ? -1 : input_shape[3] * block_width - crop_left - crop_right;
            output_shape[1] = input_shape[1] < 0 ? -1 : input_shape[1];

            return {x.dtype(), output_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "batch_to_space4d", batch_to_space4d)

        static TensorPrototype space_to_batch4d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto padding = node->get_int_list("padding");
            auto block_shape = node->get_int_list("block_shape");

            if (padding.size() < 4 || block_shape.size() < 2) return VOID;

            auto block_height = block_shape[0];
            auto block_width = block_shape[1];
            auto padding_top = padding[0];
            auto padding_bottom = padding[1];
            auto padding_left = padding[2];
            auto padding_right = padding[3];

            auto &input_shape = x.sizes();
            std::vector<int32_t> output_shape(4, -1);


            output_shape[0] = input_shape[0] < 0 ? -1 : input_shape[0] * block_height * block_width;
            output_shape[2] = input_shape[2] < 0 ? -1 : (input_shape[2] + padding_top + padding_bottom) / block_height;
            output_shape[3] = input_shape[3] < 0 ? -1 : (input_shape[3] + padding_left + padding_right) / block_width;
            output_shape[1] = input_shape[1] < 0 ? -1 : input_shape[1];

            return {x.dtype(), output_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "space_to_batch4d", space_to_batch4d)

        static TensorPrototype _field(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto offset = node->get_int("offset");

            if (offset < 0) offset += int32_t(x.fields_count());
            if (offset < 0 || offset >= int32_t(x.fields_count())) return VOID;

            return x.field(offset);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_field", _field)

        static TensorPrototype _pack(const Node &node, const std::vector<TensorPrototype> &inputs) {
            std::vector<Tensor::Prototype> output;
            for (auto &i : inputs) {
                output.emplace_back(i.dtype(), i.sizes());
            }
            TensorPrototype packed;
            packed.pack(output);
            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "_pack", _pack)

        static TensorPrototype quantize(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            return {INT8, x.sizes()};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "quantize", quantize)

        static TensorPrototype broadcast(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto dtype = inputs[0].dtype();

            auto lhs_shape = inputs[0].sizes();

            auto rhs_shape_value = get_value(node.input(1));
            if (rhs_shape_value.empty()) return VOID;
            Shape rhs_shape = tensor::array::to_int(rhs_shape_value);

            if (lhs_shape.size() > rhs_shape.size()) {
                begin_insert_ones(rhs_shape, lhs_shape.size() - rhs_shape.size());
            } else if (rhs_shape.size() > lhs_shape.size()) {
                begin_insert_ones(lhs_shape, rhs_shape.size() - lhs_shape.size());
            }

            auto dims = lhs_shape.size();
            auto out_shape = std::vector<int32_t>(dims, -1);

            for (size_t i = 0; i < dims; ++i) {
                out_shape[i] = eltwise_infer_dim(lhs_shape[i], rhs_shape[i]);
            }

            return {dtype, out_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "broadcast", broadcast)

        static TensorPrototype chunk(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto chunks = node->get_int("chunks");
            auto dim = node->get_int("dim");

            if (dim < 0) dim += int32_t(x.dims());
            if (dim < 0 || dim >= int32_t(x.dims())) return VOID;

            auto bins = split_bins(0, x.size(dim), chunks);

            std::vector<Tensor::Prototype> output;
            for (auto &bin : bins) {
                auto tmp = x.sizes();
                tmp[dim] = bin.second - bin.first;
                output.emplace_back(x.dtype(), tmp);
            }

            TensorPrototype packed;
            packed.pack(output);

            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "chunk", chunk)

        static TensorPrototype conv2d_quantized(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {FLOAT32, y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d_quantized", conv2d_quantized)

        static TensorPrototype conv2d_winograd(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            std::vector<int32_t> stide = {1, 1, 1, 1};
            std::vector<int32_t> dilation = {1, 1, 1, 1};
            std::vector<int32_t> padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {3, 3};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d_winograd", conv2d_winograd)

        static TensorPrototype conv2d_winograd_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            auto padding = tensor::array::to_int(padding_value);

            std::vector<int32_t> stide = {1, 1, 1, 1};
            std::vector<int32_t> dilation = {1, 1, 1, 1};

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[2];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {3, 3};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#padding", tensor::build<int32_t>(INT32, {4, 2}, padding));
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "conv2d_winograd_v2", conv2d_winograd_v2)

        static TensorPrototype dcn_v2_forward(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "dcn_v2_forward", dcn_v2_forward)

        static TensorPrototype depthwise_conv2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0) * x.size(channel_dims);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "depthwise_conv2d", depthwise_conv2d)

        static TensorPrototype depthwise_conv2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto padding_value = get_value(node.input(1));
            auto padding = tensor::array::to_int(padding_value);

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[2];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(0) * x.size(channel_dims);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_forward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#padding", tensor::build<int32_t>(INT32, {4, 2}, padding));
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "depthwise_conv2d_v2", depthwise_conv2d_v2)

        static TensorPrototype divided(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto size = node->get_int_list("size");

            if (size.size() > x.dims()) return VOID;

            while (size.size() < x.dims()) {
                size.insert(size.begin(), 1);
            }

            auto y_shape = x.sizes();

            for (size_t i = 0; i < x.dims(); ++i) {
                if (size[i] == 1) continue;
                y_shape[i] = int32_t(std::ceil(float(y_shape[i]) / size[i])) * size[i];
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "divided", divided)

        static TensorPrototype force_color(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            if (x.dims() == 0) return VOID;

            auto y_shape = x.sizes();
            y_shape.back() = 3;

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "force_color", force_color)

        static TensorPrototype force_gray(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            if (x.dims() == 0) return VOID;

            auto y_shape = x.sizes();
            y_shape.back() = 1;

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "force_gray", force_gray)

        static TensorPrototype gather(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            auto &indices = inputs[1].sizes();
            auto dims = int32_t(x.dims());

            auto axis = node->get_int("axis");

            if (axis < 0) axis += dims;
            if (axis < 0 || axis >= dims) return VOID;

            auto y_shape = x.sizes();

            y_shape.erase(y_shape.begin() + axis);
            y_shape.insert(y_shape.begin() + axis, indices.begin(), indices.end());

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "gather", gather)

        static TensorPrototype gatherv2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];
            auto &indices = inputs[1];

            if (x.dims() == 0 || indices.dims() == 0) return VOID;

            Shape output_shape = indices.sizes();
            output_shape.erase(output_shape.end() - 1);

            auto &indices_shape = indices.sizes();
            auto input_shape = x.sizes();
            if (indices_shape[indices_shape.size() - 1] > input_shape.size()) return VOID;

            output_shape.insert(output_shape.end(), input_shape.begin() + indices_shape[indices_shape.size() - 1],
                                input_shape.end());

            return {x.dtype(), output_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "gatherv2", gatherv2)

        class ReductionOp {
        public:
            std::string axes_attr_name;
            std::string keep_axes_attr_name;

            ReductionOp(const std::string &axes_attr, const std::string &keep_axes_attr) TS_NOEXCEPT
                    : axes_attr_name(axes_attr), keep_axes_attr_name(keep_axes_attr) {}

            TensorPrototype operator()(const Node &node, const std::vector<TensorPrototype> &inputs) {
                auto &x = inputs[0];
                auto dims = int32_t(x.dims());

                auto axes = node->get_int_list(axes_attr_name);
                auto keep_axes = node->get_bool(keep_axes_attr_name);

                for (auto &axis : axes) {
                    if (axis < 0) axis += dims;
                    if (axis < 0 || axis >= dims) return VOID;
                }

                std::sort(axes.begin(), axes.end(), [](int a, int b) { return a > b; });

                auto y_shape = x.sizes();
                for (auto axis : axes) {
                    y_shape.erase(y_shape.begin() + axis);
                    if (keep_axes) {
                        y_shape.insert(y_shape.begin() + axis, 1);
                    }
                }

                return {x.dtype(), y_shape};
            }
        };

        TS_STATIC_ACTION(ShapeInferer::Register, "max", ReductionOp("dim", "keep_dims"))

        TS_STATIC_ACTION(ShapeInferer::Register, "maximum", _eltwise)

        static TensorPrototype non_max_suppression_v3(const Node &node, const std::vector<TensorPrototype> &inputs) {
            // auto &x = inputs[0];
            auto &scores = inputs[1];

            if (scores.dims() == 0) return VOID;

            auto max_output_size = node->get_int("max_output_size");

            auto K = std::min(scores.size(0), max_output_size);

            return {INT32, {K,}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "non_max_suppression_v3", non_max_suppression_v3)

        static int32_t conv2d_backward(int32_t y, int32_t padding, int32_t dilation, int32_t kernel, int32_t stride) {
            return (y - 1) * stride + (dilation * (kernel - 1) + 1) - padding;
        }

        static TensorPrototype transpose_conv2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto format = node->get_string("format");

            auto stide = node->get_int_list("stride");
            auto dilation = node->get_int_list("dilation");
            auto padding = node->get_int_list("padding");

            std::vector<int32_t> kernel_dims;
            int32_t channel_dims;

            auto &x = inputs[0];
            auto &w = inputs[1];

            if (format == "NCHW") {
                channel_dims = 1;
                kernel_dims = {2, 3};
            } else if (format == "NHWC") {
                channel_dims = 3;
                kernel_dims = {1, 2};
            } else {
                return VOID;
            }

            std::vector<int32_t> y_shape(4);
            y_shape[0] = x.size(0);
            y_shape[channel_dims] = w.size(1);

            int32_t kernel_shape[] = {w.size(2), w.size(3)};

            for (size_t i = 0; i < kernel_dims.size(); ++i) {
                auto dim = kernel_dims[i];
                if (x.size(dim) < 0) {
                    y_shape[dim] = -1;
                    continue;
                }
                y_shape[dim] = conv2d_backward(
                        x.size(dim),
                        padding[2 * dim] + padding[2 * dim + 1],
                        dilation[dim],
                        kernel_shape[i],
                        stide[dim]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "transpose_conv2d", transpose_conv2d)

        static TensorPrototype winograd_transform_kernel(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto winograd_mode_str = node->get_string("winograd_mode");

            enum WINOGRAD_MODE {
                F6X6_3X3 = 0,
                F2X2_3X3 = 1,
            };

            WINOGRAD_MODE winograd_mode;

            if (winograd_mode_str == "winograd_f23") {
                winograd_mode = F2X2_3X3;
            } else if (winograd_mode_str == "winograd_f63") {
                winograd_mode = F6X6_3X3;
            } else {
                return VOID;
            }

            auto y_shape = x.sizes();

            if (winograd_mode == F6X6_3X3) {
                y_shape[2] = 8;
                y_shape[3] = 8;
            } else if (winograd_mode == F2X2_3X3) {
                y_shape[2] = 4;
                y_shape[3] = 4;
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "winograd_transform_kernel", winograd_transform_kernel)

        static TensorPrototype pad(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            auto padding_value = get_value(node.input(1));
            if (padding_value.empty()) return VOID;
            auto padding = tensor::array::to_int(padding_value);

            if (padding.size() != x.dims() * 2) return VOID;

            auto y_shape = x.sizes();
            for (size_t i = 0; i < x.dims(); ++i) {
                if (y_shape[i] < 0) continue;
                y_shape[i] += padding[i * 2] + padding[i * 2 + 1];
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "pad", pad)

#define TYR_GET_NODE_INT_ATTR(attr, value) \
    int32_t attr = value; \
    if (node->has(#attr)) attr = node->get_int(#attr);

        static TensorPrototype proposal(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            if (inputs.size() < 3) return VOID;
            auto dtype = inputs[inputs.size() - 3].dtype();

            TYR_GET_NODE_INT_ATTR(min_level, 2)
            TYR_GET_NODE_INT_ATTR(max_level, 5)
            TYR_GET_NODE_INT_ATTR(post_nms_top_n, 300)

            auto num_images = x.size(0);
            auto output_size = max_level - min_level + 1;

            std::vector<Tensor::Prototype> output;
            for (int i = 0; i < output_size; ++i) {
                Shape tmp = {num_images > 0 ? num_images * post_nms_top_n : -1, 5};
                output.emplace_back(dtype, tmp);
            }

            TensorPrototype packed;
            packed.pack(output);

            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "proposal", proposal)

        static TensorPrototype range(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.size() != 3) return VOID;

            auto output = try_run(node, {
                get_value(node.input(0)), get_value(node.input(1)),get_value(node.input(2)), });

            if (output.empty()) return VOID;

            {
                auto &update_node = *const_cast<Node*>(&node);
                update_node->set("#value", output);
            }

            return TensorPrototype(output);
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "range", range)

        TS_STATIC_ACTION(ShapeInferer::Register, "reduce_mean", ReductionOp("dims", "keep_dims"))
        TS_STATIC_ACTION(ShapeInferer::Register, "reduce_sum", ReductionOp("dims", "keep_dims"))

#define GET_NODE_INT_ATTR(attr) \
    if (!node->has(#attr)) return VOID; \
    auto attr = node->get_int(#attr);

#define GET_NODE_FLOAT_ATTR(attr) \
    if (!node->has(#attr)) return VOID; \
    auto attr = node->get_float(#attr);

#define GET_NODE_INT_LIST_ATTR(attr) \
    if (!node->has(#attr)) return VOID; \
    auto attr = node->get_int_list(#attr);

#define GET_NODE_OTL_INT_LIST_ATTR(attr) \
    if (!node->has(#attr)) return VOID; \
    Shape attr = node->get_int_list(#attr);

#define TRY_GET_NODE_INT_LIST_ATTR(attr) \
    std::vector<int32_t> attr; \
    if (node->has(#attr)) \
        attr = node->get_int_list(#attr);

#define TRY_GET_NODE_INT_ATTR(attr, value) \
    int32_t attr = value; \
    if (node->has(#attr)) \
        attr = node->get_int(#attr);

#define GET_NODE_FLOAT_LIST_ATTR(attr) \
    if (!node->has(#attr)) return VOID; \
    auto attr = node->get_float_list(#attr);

#define GET_NODE_INT_LIST_INPUT(var, i) \
    std::vector<int32_t> var; \
    { \
        if (size_t(i) >= node.inputs().size()) return VOID; \
        auto tmp = get_value(node.input(i)); \
        if (tmp.empty()) return VOID; \
        var = tensor::array::to_int(tmp); \
    }

#define TRY_GET_NODE_INT_LIST_INPUT(var, i) \
    std::vector<int32_t> var; \
    { \
        if (size_t(i) < node.inputs().size()) { \
            auto tmp = get_value(node.input(i)); \
            if (tmp.empty()) return VOID; \
            var = tensor::array::to_int(tmp); \
        } \
    }

#define TRY_GET_NODE_FLOAT_LIST_INPUT(var, i) \
    std::vector<float> var; \
    { \
        if (size_t(i) < node.inputs().size()) { \
            auto tmp = get_value(node.input(i)); \
            if (tmp.empty()) return VOID; \
            var = tensor::array::to_float(tmp); \
        } \
    }

#define FIX_DIM(dim, x) \
    if (dim < 0) dim += int32_t(x.dims()); \
    if (dim < 0 || dim >= int32_t(x.dims())) return VOID;

#define FIX_DIM_ADD_1(dim, x) \
    if (dim < 0) dim += int32_t(x.dims()); \
    if (dim < 0 || dim + 1 >= int32_t(x.dims())) return VOID;

        static TensorPrototype resize_nearest_neighbor(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            GET_NODE_INT_LIST_INPUT(size, 1)
            GET_NODE_INT_ATTR(dim)

            if (size.size() < 2) return VOID;

            FIX_DIM_ADD_1(dim, x)

            auto y_shape = x.sizes();
            y_shape[dim] = size[0];
            y_shape[dim + 1] = size[1];

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "resize_nearest_neighbor", resize_nearest_neighbor)

        static TensorPrototype roi_align(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.size() != 2) return VOID;

            GET_NODE_INT_ATTR(pool_h)
            GET_NODE_INT_ATTR(pool_w)

            // inputs = features, proposal

            return {inputs[0].dtype(), {inputs[1].size(0), inputs[0].size(1), pool_h, pool_w}};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "roi_align", roi_align)

        static TensorPrototype sample2d(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.size() != 1) return VOID;
            auto &x = inputs[0];

            GET_NODE_FLOAT_ATTR(scale)
            TYR_GET_NODE_INT_ATTR(dim, -2)

            FIX_DIM_ADD_1(dim, x)

            auto y_shape = x.sizes();
            if (y_shape[dim] > 0)
                y_shape[dim] = int32_t(x.sizes()[dim] * scale);
            if (y_shape[dim + 1] > 0)
                y_shape[dim + 1] = int32_t(x.sizes()[dim + 1] * scale);

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "sample2d", sample2d)

        static TensorPrototype shape_index_patch(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.size() != 1) return VOID;
            auto &x = inputs[0];
            auto &pos = inputs[1];

            auto number = x.size(0);
            auto channels = x.size(1);
            auto height = x.size(2);
            auto width = x.size(3);
            auto landmarkx2 = pos.size(1);

            GET_NODE_INT_LIST_ATTR(origin_patch)
            GET_NODE_INT_LIST_ATTR(origin)

            auto x_patch_h = int(origin_patch[0] * height / float(origin[0]) + 0.5f);
            auto x_patch_w = int(origin_patch[1] * width / float(origin[1]) + 0.5f);

            Shape y_shape = {number, channels, x_patch_h, landmarkx2 / 2, x_patch_w};

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "shape_index_patch", shape_index_patch)

        static TensorPrototype squeeze(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            GET_NODE_INT_LIST_ATTR(axes)

            auto y_shape = x.sizes();

            if (axes.empty()) {
                for (auto it = axes.begin(); it != axes.end(); ) {
                    if (*it == 1) {
                        it = axes.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else {
                for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
                    auto axis = *it;
                    FIX_DIM(axis, x)
                    y_shape.erase(y_shape.begin() + axis);
                }
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "squeeze", squeeze)

        static TensorPrototype unsqueeze(const Node &node, const std::vector<TensorPrototype> &inputs) {
            auto &x = inputs[0];

            GET_NODE_INT_LIST_ATTR(axes)

            auto y_shape = x.sizes();

            for (auto axis : axes) {
                auto max_axis = int32_t(y_shape.size());
                if (axis > max_axis || axis < -max_axis) {
                    return VOID;
                }
                if (axis >= 0) {
                    y_shape.insert(y_shape.begin() + axis, 1);
                } else {
                    y_shape.insert(y_shape.end() + axis + 1, 1);
                }
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "unsqueeze", unsqueeze)

        static TensorPrototype stack(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;
            auto &x = inputs[0];

            GET_NODE_INT_ATTR(axis)

            auto y_shape = x.sizes();

            if (axis >= 0) {
                y_shape.insert(y_shape.begin() + axis, int32_t(inputs.size()));
            } else {
                y_shape.insert(y_shape.end() + axis + 1, int32_t(inputs.size()));
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "stack", stack)

        static Shape infer_tile(Shape &x, Shape &repeats) {
            if (x.size() == repeats.size()) {
            } else if (x.size() > repeats.size()) {
                do {
                    repeats.insert(repeats.begin(), 1);
                } while (x.size() > repeats.size());
            } else{
                do {
                    x.insert(x.begin(), 1);
                } while (x.size() < repeats.size());
            }
            Shape y(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                if (x[i] < 0) {
                    y[i] = -1;
                    continue;
                }
                y[i] = x[i] * repeats[i];
            }
            return y;
        }

        static TensorPrototype tile(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;
            auto &x = inputs[0];

            GET_NODE_OTL_INT_LIST_ATTR(repeats)

            auto y_shape = x.sizes();

            return {x.dtype(), infer_tile(y_shape, repeats)};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "tile", tile)

        static TensorPrototype topkv2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;
            auto &x = inputs[0];

            GET_NODE_INT_ATTR(number)

            Tensor::Prototype y;
            Tensor::Prototype index;

            if (x.dims() == 0) {
                y = Tensor::Prototype(x.dtype(), x.sizes());
                index = Tensor::Prototype(INT32, x.sizes());
            } else {
                auto K = std::min(x.sizes().back(), number);
                if (K < 0) K = number;
                auto y_shape = x.sizes();
                y_shape.back() = K;
                y = Tensor::Prototype(x.dtype(), y_shape);
                index = Tensor::Prototype(INT32, y_shape);
            }

            TensorPrototype packed;
            packed.pack({y, index});

            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "topkv2", topkv2)

        static TensorPrototype yolo(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &x = inputs[0];
            if (x.dims() != 4) return VOID;

            GET_NODE_INT_ATTR(classes)
            GET_NODE_INT_LIST_ATTR(mask)
            GET_NODE_FLOAT_LIST_ATTR(anchors)

            auto n = int32_t(mask.size());

            auto batch = x.size(0);
            auto h = x.size(2);
            auto w = x.size(3);

            std::vector<Tensor::Prototype> outputs = {
                    Tensor::Prototype(x.dtype(), {batch, n * (classes + 4 + 1), h, w}),
                    Tensor::Prototype(INT32, Shape()),
                    Tensor::Prototype(INT32, {int32_t(mask.size())}),
                    Tensor::Prototype(FLOAT32, {int32_t(anchors.size())}),

            };

            TensorPrototype packed;
            packed.pack(outputs);
            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "yolo", yolo)

        static TensorPrototype yolo_poster(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &yolo = inputs.back();

            auto N = yolo.size(0);
            if (N < 0) N = 1;

            std::vector<Tensor::Prototype> outputs(N, Tensor::Prototype(FLOAT32, {-1, 6}));

            TensorPrototype packed;
            packed.pack(outputs);
            return packed;
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "yolo_poster", yolo_poster)

        static TensorPrototype slice(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &x = inputs[0];

            GET_NODE_INT_LIST_ATTR(begin)
            GET_NODE_INT_LIST_ATTR(size)

            if (begin.size() > x.dims()) return VOID;
            if (begin.size() != size.size()) return VOID;

            auto y_shape = x.sizes();
            for (size_t i = 0; i < size.size(); ++i) {
                if (y_shape[i] < 0) {
                    y_shape[i] = size[i];
                    continue;
                }
                if (begin[i] >= y_shape[i]) {
                    y_shape[i] = 0;
                    continue;
                }
                auto right = std::min(y_shape[i], begin[i] + size[i]);
                y_shape[i] = right - begin[i];
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "slice", slice)

        static int infer_output(int x, int &begin, int &end, int stride, bool begin_flag, bool end_flag) {
            // begin \in [0, x)
            // end \in [-1, x]
            if (x >= 0) {
                if (begin_flag) {
                    begin = stride > 0 ? 0 : x - 1;
                } else {
                    if (stride > 0) {
                        if (begin >= x) return 0;  // no elements
                        else if (begin < -x) begin = 0;
                        else if (begin < 0) begin += x;
                    } else {
                        if (begin < -x) return 0;  // no elements
                        else if (begin >= x) begin = x - 1;
                        else if (begin < 0) begin += x;
                    }
                }
                if (end_flag) {
                    end = stride > 0 ? x : -1;
                } else {
                    if (stride > 0) {
                        if (end <= -x) return 0;     // no elements
                        else if (end > x) end = x;
                        else if (end < 0) end += x;
                    } else {
                        if (end > x) return 0;     // no elements
                        else if (end <= -x) end = -1;
                        else if (end < 0) end += x;
                    }
                }
            }

            if (stride > 0) {
                return begin < end ? (end - begin - 1) / stride + 1 : 0;
            } else if (stride < 0) {
                return begin > end ? (begin - end - 1) / -stride + 1 : 0;
            } else {
                return -1;
            }
        }

        static TensorPrototype slice_v3(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &x = inputs[0];

            GET_NODE_INT_LIST_INPUT(starts, 1)
            GET_NODE_INT_LIST_INPUT(ends, 2)
            TRY_GET_NODE_INT_LIST_INPUT(axes, 3)
            TRY_GET_NODE_INT_LIST_INPUT(steps, 4)

            if (starts.size() != ends.size()) return VOID;

            if (axes.empty()) {
                for (size_t i = 0; i < starts.size(); ++i) {
                    axes.push_back(int32_t(i));
                }
            } else {
                if (starts.size() != axes.size()) return VOID;
            }

            if (steps.empty()) {
                steps = std::vector<int32_t>(starts.size(), 1);
            } else {
                if (starts.size() != steps.size()) return VOID;
            }

            auto y_shape = x.sizes();
            auto N = starts.size();

            for (size_t i = 0; i < N; ++i) {
                auto dim = axes[i];

                y_shape[dim] = infer_output(
                        y_shape[dim],
                        starts[i],
                        ends[i],
                        steps[i],
                        false,
                        false);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "slice_v3", slice_v3)

        class SliceDim {
        public:
            SliceDim() = default;

            SliceDim(int begin, int end, int stride = 1,
                    bool begin_flag = false, bool end_flag = false,
                    bool ellipsis_flag = false,
                    bool new_axis_flag = false, bool shrink_axis_flag = false) {
                this->begin = begin;
                this->end = end;
                this->stride = stride;
                this->flag.begin = begin_flag;
                this->flag.end = end_flag;
                this->flag.ellipsis = ellipsis_flag;
                this->flag.new_axis = new_axis_flag;
                this->flag.shrink_axis = shrink_axis_flag;
            }

            int begin = 0;
            int end = 0;
            int stride = 1;
            struct {
                bool begin = false;
                bool end = false;
                bool ellipsis = false;
                bool new_axis = false;
                bool shrink_axis = false;
            } flag;
        };

        static TensorPrototype strided_slice(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &x = inputs[0];

            GET_NODE_INT_LIST_ATTR(begin)
            GET_NODE_INT_LIST_ATTR(end)
            TRY_GET_NODE_INT_LIST_ATTR(stride)
            TRY_GET_NODE_INT_ATTR(begin_mask, 0)
            TRY_GET_NODE_INT_ATTR(end_mask, 0)
            TRY_GET_NODE_INT_ATTR(ellipsis_mask, 0)
            TRY_GET_NODE_INT_ATTR(new_axis_mask, 0)
            TRY_GET_NODE_INT_ATTR(shrink_axis_mask, 0)

            if (stride.empty()) {
                stride.resize(begin.size(), 1);
            } else {
                if (begin.size() != stride.size()) return VOID;
            }
            if (begin.size() != end.size()) return VOID;

            std::vector<SliceDim> slice_dims;

            for (size_t i = 0; i < begin.size(); ++i) {
                slice_dims.emplace_back(begin[i], end[i], stride[i],
                                        begin_mask & (1 << i),
                                        end_mask & (1 << i),
                                        ellipsis_mask & (1 << i),
                                        new_axis_mask & (1 << i),
                                        shrink_axis_mask & (1 << i));
            }

            auto base_slice_dims = slice_dims;

            // deal ellipsis
            {
                // find ellipsis count;
                int32_t ellipsis_count = 0;
                int32_t sliced_count = 0;
                for (auto &slice_dim : slice_dims) {
                    if (slice_dim.flag.ellipsis) {
                        ++ellipsis_count;
                    } else if(!slice_dim.flag.new_axis) {
                        ++sliced_count;
                    }
                }
                if (ellipsis_count > 1) return VOID;
                if (sliced_count > int32_t(x.dims())) return VOID;
                if (ellipsis_count) {
                    for (auto it = slice_dims.begin(); it != slice_dims.end(); ++it) {
                        if (it->flag.ellipsis) {
                            it = slice_dims.erase(it);
                            std::vector<SliceDim> full_select_dims(x.dims() - sliced_count,
                                                                   SliceDim(0, 0, 1, true, true));
                            slice_dims.insert(it, full_select_dims.begin(), full_select_dims.end());
                            break;
                        }
                    }
                } else if (sliced_count < int32_t(x.dims())) {
                    std::vector<SliceDim> full_select_dims(x.dims() - sliced_count,
                                                           SliceDim(0, 0, 1, true, true));
                    slice_dims.insert(slice_dims.end(), full_select_dims.begin(), full_select_dims.end());
                }
            }

            auto have_new_axes_slice_dims = slice_dims;

            // remove new axes
            {
                for (auto it = slice_dims.begin(); it != slice_dims.end(); ) {
                    if (it->flag.new_axis) {
                        it = slice_dims.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            if (slice_dims.size() != x.dims()) return VOID;

            auto y_shape = x.sizes();
            for (size_t i = 0; i < slice_dims.size(); ++i) {
                auto &s = slice_dims[i];
                y_shape[i] = infer_output(y_shape[i], s.begin, s.end, s.stride, s.flag.begin, s.flag.end);
            }

            auto fixed_value_slice_dims = slice_dims;
            // do new_axis
            slice_dims = have_new_axes_slice_dims;
            {
                for (size_t i = 0; i < slice_dims.size(); ++i) {
                    auto &s = slice_dims[i];
                    if (s.flag.new_axis) {
                        y_shape.insert(y_shape.begin() + i, 1);
                    }
                }
            }
            // do shrink axis
            {
                for (int i = int(slice_dims.size()) - 1; i >= 0; --i) {
                    auto &s = slice_dims[i];
                    if (s.flag.shrink_axis) {
                        y_shape.erase(y_shape.begin() + i);
                    }
                }
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "strided_slice", strided_slice)

        TS_STATIC_ACTION(ShapeInferer::Register, "leaky_relu", _copy)

        static TensorPrototype sample2d_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.size() != 2) return VOID;
            auto &x = inputs[0];
            TRY_GET_NODE_FLOAT_LIST_INPUT(scale, 1)

            if (x.dims() != int(scale.size())) return VOID;

            auto y_shape = x.sizes();
            for (size_t i = 0; i < y_shape.size(); ++i) {
                y_shape[i] = int32_t(float(x.size(i)) * scale[i]);
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "sample2d_v2", sample2d_v2)

        static TensorPrototype slice_v2(const Node &node, const std::vector<TensorPrototype> &inputs) {
            if (inputs.empty()) return VOID;

            auto &x = inputs[0];

            TRY_GET_NODE_INT_LIST_INPUT(begin, 1)
            TRY_GET_NODE_INT_LIST_INPUT(size, 2)

            if (begin.size() > x.dims()) return VOID;
            if (begin.size() != size.size()) return VOID;

            auto y_shape = x.sizes();
            for (size_t i = 0; i < size.size(); ++i) {
                if (y_shape[i] < 0) {
                    y_shape[i] = size[i];
                    continue;
                }
                if (begin[i] >= y_shape[i]) {
                    y_shape[i] = 0;
                    continue;
                }
                auto right = std::min(y_shape[i], begin[i] + size[i]);
                y_shape[i] = right - begin[i];
            }

            return {x.dtype(), y_shape};
        }

        TS_STATIC_ACTION(ShapeInferer::Register, "slice_v2", slice_v2)
    }
}
