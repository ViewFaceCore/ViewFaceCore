//
// Created by kier on 2018/10/31.
//

#include <module/menu.h>

#include "module/menu.h"
#include "core/tensor_builder.h"

#include "backend/name.h"

namespace ts {
    namespace bubble {
        Node param(const std::string &name) {
            auto &g = ctx::ref<Graph>();
            return g.make(Bubble::Parameter, name);
        }

        Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs) {
            auto &g = ctx::ref<Graph>();
            Node result = g.make(op_name, name);
            Node::Link(result, inputs);
            return result;
        }

        Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs, int output_count) {
            TS_AUTO_CHECK(output_count == 1);
            return op(name, op_name, inputs);
        }

        Node data(const std::string &name, const Tensor &value) {
            auto &g = ctx::ref<Graph>();
            Node result = g.make(Bubble::Const, name);
            result->set(name::value, value);
            return result;
        }

        Node data(const std::string &name, const Tensor &value, const DeviceType &device) {
            auto &g = ctx::ref<Graph>();
            Node result = g.make(Bubble::Const, name);
            result->set(name::value, value);
            result->set(name::device, tensor::from(device));
            return result;
        }

        Node param(const std::string &name, const Shape &shape) {
            auto &g = ctx::ref<Graph>();
            auto result = g.make(Bubble::Parameter, name, shape);
            return result;
        }

        Node param(const std::string &name, DTYPE dtype) {
            auto &g = ctx::ref<Graph>();
            auto result = g.make(Bubble::Parameter, name);
            result->set(Bubble::RetentionParam::dtype, tensor::from<int32_t>(dtype));
            return result;
        }

        Node param(const std::string &name, DTYPE dtype, const Shape &shape) {
            auto &g = ctx::ref<Graph>();
            auto result = g.make(Bubble::Parameter, name, shape);
            result->set(Bubble::RetentionParam::dtype, tensor::from<int32_t>(dtype));
            return result;
        }

        Node bubble(const Bubble &bubble) {
            auto &g = ctx::ref<Graph>();
            return g.make(bubble);
        }

        Node bubble(const Bubble &bubble, const std::string &name) {
            auto &g = ctx::ref<Graph>();
            auto result = g.make(bubble);
            result->name(name);
            return result;
        }
    }

    size_t serialize_graph(StreamWriter &stream, const Graph &graph) {
        return serialize_nodes(stream, graph.nodes(), 0);
    }

    size_t externalize_graph(StreamReader &stream, Graph &graph) {
        ctx::bind<Graph> _bind_graph(graph);
        return externalize_nodes(stream);
    }

    size_t serialize_nodes(StreamWriter &stream, const std::vector<Node> &nodes, size_t base) {
        std::unordered_map<Node, size_t> map_node_index;
        // build node index
        size_t index = base;
        for (auto &node : nodes) {
            map_node_index.insert(std::make_pair(node, index++));
        }
        //
        size_t writen_size = 0;
        std::vector<size_t> node_input_index;
        // 1.0 write size
        writen_size += binio::write<uint32_t>(stream, uint32_t(nodes.size()));
        // write each node
        for (auto &node : nodes) {
            node_input_index.clear();
            for (auto &input : node.inputs()) {
                auto input_it = map_node_index.find(input);
                if (input_it == map_node_index.end()) {
                    TS_LOG_ERROR << "Can not link input " << input << " in node " << node << eject;
                }
                node_input_index.push_back(input_it->second);
            }
            // 0.1 write bubble
            auto &bubble = node.bubble();
            writen_size += bubble.serialize(stream);
            // 0.2 write input index
            writen_size += binio::write<uint32_t>(stream, uint32_t(node_input_index.size()));
            for (auto &input_index : node_input_index) {
                writen_size += binio::write<uint32_t>(stream, uint32_t(input_index));
            }
        }
        return writen_size;
    }

    size_t externalize_nodes(StreamReader &stream) {
        auto &g = ctx::ref<Graph>();

        using NodeWithInputIndex = std::pair<Node, std::vector<uint32_t>>;
        size_t read_size = 0;
        std::vector<NodeWithInputIndex> nodes;
        uint32_t node_count = 0;
        uint32_t size_buffer = 0;
        // 1. read size
        read_size += binio::read<uint32_t>(stream, node_count);
        for (uint32_t i = 0; i < node_count; ++i) {
            // .1 read bubble
            Bubble bubble;
            read_size += bubble.externalize(stream);
            // .2 read index
            read_size += binio::read<uint32_t>(stream, size_buffer);
            std::vector<uint32_t> node_input_index(size_buffer);
            for (auto &input_index : node_input_index) {
                read_size += binio::read<uint32_t>(stream, input_index);
            }
            nodes.emplace_back(std::make_pair(g.make(bubble), node_input_index));
        }
        // 2. link nodes
        for (auto &node : nodes) {
            std::vector<Node> inputs;
            for (auto index : node.second) {
                inputs.emplace_back(nodes[index].first);
            }
            Node::Link(node.first, inputs);
        }
        return read_size;
    }
}