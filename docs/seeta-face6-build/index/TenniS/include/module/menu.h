//
// Created by kier on 2018/10/31.
//

#ifndef TENSORSTACK_MODULE_MENU_H
#define TENSORSTACK_MODULE_MENU_H

#include "module.h"
#include "utils/ctxmgr.h"

namespace ts {
    namespace bubble {
        /**
         * get Parameter node
         * @param name Node name
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node param(const std::string &name);

        /**
         * get Parameter node
         * @param name Node name
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node param(const std::string &name, const Shape &shape);

        /**
         * get Parameter node
         * @param name Node name
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node param(const std::string &name, DTYPE dtype);

        /**
         * get Parameter node
         * @param name Node name
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node param(const std::string &name, DTYPE dtype, const Shape &shape);

        /**
         * get Operator node
         * @param name Node name
         * @param op_name Operator name
         * @param inputs Input nodes
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs);

        /**
         * get Operator node
         * @param name Node name
         * @param op_name Operator name
         * @param inputs Input nodes
         * @param output_count Output count, must be 1.
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node op(const std::string &name, const std::string &op_name, const std::vector<Node> &inputs, int output_count);

        /**
         * get Data node
         * @param name Node name
         * @param value the data value
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node data(const std::string &name, const Tensor &value);

        /**
         * get Data node
         * @param name Node name
         * @param value the data value
         * @param device the memory saving device
         * @return new Node belonging to context-Graph
         * @note Must call `ts::ctx::bind<Graph>` to bind context firstly
         */
        TS_DEBUG_API Node data(const std::string &name, const Tensor &value, const DeviceType &device);

        TS_DEBUG_API Node bubble(const Bubble &bubble);

        TS_DEBUG_API Node bubble(const Bubble &bubble, const std::string &name);
    }

    /**
     * write nodes to file, ref as nodes
     * @param stream writer
     * @param nodes nodes ready to save to stream
     * @param base node index base in graph
     * @return writen size
     * @note do not parse param `base` in this version
     */
    TS_DEBUG_API size_t serialize_nodes(StreamWriter &stream, const std::vector<Node> &nodes, size_t base = 0);

    /**
     * you need call ctx::bind<Graph> first
     * @param stream reader
     * @return read size
     */
    TS_DEBUG_API size_t externalize_nodes(StreamReader &stream);

    /**
     * @param stream writer
     * @param graph ready write graph
     * @return writen size
     */
    TS_DEBUG_API size_t serialize_graph(StreamWriter &stream, const Graph &graph);

    /**
     * @param stream reader
     * @param graph ready read graph
     * @return read size
     */
    TS_DEBUG_API size_t externalize_graph(StreamReader &stream, Graph &graph);
}


#endif //TENSORSTACK_MODULE_MENU_H
