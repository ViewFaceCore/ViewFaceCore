//
// Created by kier on 2019/11/20.
//

#ifndef TENSORSTACK_RUNTIME_INFERER_H
#define TENSORSTACK_RUNTIME_INFERER_H

#include "module/graph.h"

namespace ts {
    TS_DEBUG_API TensorPrototype infer(Node &node, std::unordered_map<Node, TensorPrototype> &cache);

    TS_DEBUG_API std::vector<TensorPrototype> infer(std::vector<Node> &nodes, std::unordered_map<Node, TensorPrototype> &cache);

    TS_DEBUG_API TensorPrototype infer(Node &node);

    TS_DEBUG_API std::vector<TensorPrototype> infer(std::vector<Node> &nodes);

    /**
     * set #value attr if the value can be inferred with all input nodes inferred
     * @param node node ready to infer
     * @note all input attr #shape must be needed
     * @note if ignore is empty, mean all inputs can not be ignored
     * @note <const> Node happen nothing
     */
    TS_DEBUG_API void infer_value(Node &node);
}

#endif // TENSORSTACK_RUNTIME_INFERER_H