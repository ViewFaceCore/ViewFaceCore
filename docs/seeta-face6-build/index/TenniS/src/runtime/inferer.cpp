//
// Created by kier on 2019/11/20.
//

#include <core/tensor_builder.h>
#include <core/device_context.h>
#include "runtime/inferer.h"
#include "global/shape_inferer_factory.h"

#include "runtime/stack.h"
#include "global/operator_factory.h"

#include "global/hard_allocator.h"

namespace ts {
    static Tensor get_value(const Node &node) {
        if (node->op() == Bubble::Const) {
            return node->get("value");
        }
        if (node->has("#value")) {
            return node->get("#value");
        }
        return Tensor();
    }

    void *FakeMemoryAllocator(int, size_t, void *, size_t) {
        return nullptr;
    }

    TS_STATIC_ACTION(HardAllocator::Register, "_fake_", FakeMemoryAllocator)

    // static bool valid_shape(const Shape &shape) {
    //     for (auto &dim : shape) {
    //         if (dim < 0) return false;
    //     }
    //     return true;
    // }

    void infer_value(Node &node) {
        if (Bubble::IsEndPoint(node->op())) {
            return;
        }

        std::vector<Tensor> inputs(node.inputs().size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto input = node.input(i);
            auto this_input = get_value(input);
            if (!this_input.empty()) {
                inputs[i] = this_input;
                continue;
            }
            return;
        }
        MemoryDevice memory_device(CPU);
        Stack stack(memory_device);

        DeviceContext device_context(CPU);
        ctx::bind<DeviceContext> _bind_device_context(device_context);

        auto op = OperatorCreator::CreateNoException(memory_device.type(), node->op());
        if (op == nullptr) return;

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
            return;
        }

        node->set("#value", output);
    }

    static TensorPrototype update_cache(Node &node, std::unordered_map<Node, TensorPrototype> &cache, const TensorPrototype &proto) {
        cache.insert(std::make_pair(node, proto));
        if (node->op() != Bubble::Parameter) {
            auto N = proto.fields_count();
            for (size_t i =0; i < N; ++i) {
                auto shape_key = Bubble::RetentionParam::shape + (i ? "_" + std::to_string(i) : "");
                auto dtype_key = Bubble::RetentionParam::dtype + (i ? "_" + std::to_string(i) : "");
                auto dtype_shape = proto.field(i);
                node->set(shape_key, tensor::build(INT32, dtype_shape.sizes()));
                node->set(dtype_key, tensor::build(INT32, int32_t(dtype_shape.dtype())));
            }
        }
        return proto;
    }

    TensorPrototype infer(Node &node, std::unordered_map<Node, TensorPrototype> &cache) {
        auto cache_it = cache.find(node);
        if (cache_it != cache.end()) return cache_it->second;

#define RETURN_CACHE(proto) return update_cache(node, cache, proto)

        std::vector<TensorPrototype> input_proto;
        for (auto &i : node.inputs()) {
            input_proto.emplace_back(infer(i, cache));
            if (input_proto.back().dtype() == VOID) {
                RETURN_CACHE(VOID);
            }
        }

        auto shape_infer = ShapeInferer::Query(node->op());
        if (shape_infer == nullptr) {
            TS_LOG_ERROR << "No method to infer " << node->op() << ":" << node->name();
            RETURN_CACHE(VOID);
        }

        auto before_node_value = get_value(node);

        auto output_proto = shape_infer(node, input_proto);
        if (output_proto.dtype() == VOID) {
            TS_LOG_ERROR << "Failed to infer " << node->op() << ":" << node->name();
            RETURN_CACHE(VOID);
        }

        auto after_node_value = get_value(node);

        if (after_node_value.empty() || after_node_value.data() == before_node_value.data()) {
            // if node's #value not set or not updated, then do infer
            infer_value(node);
        }

        // if (node->op() != Bubble::Const)
        //     TS_LOG_INFO << node->op() << ":" << node->name() << " => " << output_proto;

        RETURN_CACHE(output_proto);
#undef RETURN_CACHE
    }

    std::vector<TensorPrototype>
    infer(std::vector<Node> &nodes, std::unordered_map<Node, TensorPrototype> &cache) {
        std::vector<TensorPrototype> shapes;
        shapes.reserve(nodes.size());
        for (auto &node : nodes) {
            shapes.push_back(infer(node, cache));
        }
        return shapes;
    }

    TensorPrototype infer(Node &node) {
        std::unordered_map<Node, TensorPrototype> cache;
        return infer(node, cache);
    }

    std::vector<TensorPrototype> infer(std::vector<Node> &nodes) {
        std::unordered_map<Node, TensorPrototype> cache;
        return infer(nodes, cache);
    }
}
