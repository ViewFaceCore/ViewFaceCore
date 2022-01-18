//
// Created by kier on 2018/7/18.
//

#ifndef TENSORSTACK_MODULE_MODULE_H
#define TENSORSTACK_MODULE_MODULE_H

#include <memory>
#include <unordered_map>
#include "core/tensor.h"

#include "graph.h"

namespace ts {
    class TS_DEBUG_API Module {
    public:
        using self = Module;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        void load(Graph g);

        void load(Graph g, const std::vector<Node> &outputs);

        void load(Graph g, const std::vector<std::string> &outputs);

        const std::vector<Node> &inputs() const { return m_inputs; }

        const std::vector<Node> &outputs() const { return m_outputs; }

        const Node &input(int i) const { return m_inputs[i]; }

        const Node &output(int i) const { return m_outputs[i]; }

        const Node &input(size_t i) const { return m_inputs[i]; }

        const Node &output(size_t i) const { return m_outputs[i]; }

        void clear();

        void sort_inputs(const std::vector<Node> &inputs);

        void sort_inputs(const std::vector<std::string> &input_names);

        void sort_inputs(const std::initializer_list<std::string> &input_names);

        void sort_inputs(const std::initializer_list<const char *> &input_names) {
            sort_inputs(std::vector<std::string>(input_names.begin(), input_names.end()));
        }

        enum SerializationFormat {
            BINARY,
            DESCRIPTION,
        };

        static Module::shared Load(StreamReader &stream, SerializationFormat format = BINARY);
        static Module::shared Load(const std::string &filename, SerializationFormat format = BINARY);

        static void Save(StreamWriter &stream, Module::shared module, SerializationFormat format = BINARY);
        static void Save(const std::string &filename, Module::shared module, SerializationFormat format = BINARY);

        static std::vector<std::pair<Node, int>> list_reference_nodes(const std::vector<Node> &nodes);

        void set_param(const std::string &node_name, const std::string &param, const Tensor &value);

        static shared Load(Graph g);

        static shared Load(Graph g, const std::vector<Node> &outputs);

        static shared Load(Graph g, const std::vector<std::string> &outputs);

        static shared Translate(shared module, const ComputingDevice &device, const std::string &options);

        class Route {
        public:
            Route(int32_t in, int32_t in_out_slot, int32_t out, int32_t out_in_slot)
                    : in(in), in_out_slot(in_out_slot), out(out), out_in_slot() {
            }

            Route() = default;

            int32_t in = -1;            ///< input module index
            int32_t in_out_slot = -1;   ///< input module's output slot
            int32_t out = -1;           ///< output module index
            int32_t out_in_slot = -1;   ///< output module's input slot
        };

        static shared Fusion(const std::vector<shared> &submodules, const std::vector<Route> &routes);

    private:
        /**
         * @param g reference Graph
         * @param outputs output nodes
         * @return input node supporting computing outputs
         */
        static std::vector<Node> graph_walker(Graph g, const std::vector<Node> &outputs);

        std::vector<Node> m_inputs;
        std::vector<Node> m_outputs;
        std::vector<Graph> m_graphs;
    };
}




#endif //TENSORSTACK_MODULE_H
