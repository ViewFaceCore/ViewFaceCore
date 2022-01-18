//
// Created by kier on 2019/11/20.
//

#ifndef TENSORSTACK_GLOBAL_SHAPE_INFERER_FACTORY_H
#define TENSORSTACK_GLOBAL_SHAPE_INFERER_FACTORY_H

#include <functional>
#include "module/graph.h"

#include <set>

namespace ts {
    class TS_DEBUG_API ShapeInferer {
    public:
        /**
         * infer shape using node, and device type
         */
        using function = std::function<TensorPrototype(const Node &, const std::vector<TensorPrototype> &)>;

        TensorPrototype ShapeInfererFunction(const Node &node, const std::vector<TensorPrototype> &inputs);

        static function Query(const std::string &op) TS_NOEXCEPT;

        static void Register(const std::string &op, const function &inferer) TS_NOEXCEPT;

        /**
         * No details for this API, so DO NOT call it
         */
        static void Clear();

        static std::set<std::string> AllKeys() TS_NOEXCEPT;
    };
}

#endif //TENSORSTACK_GLOBAL_SHAPE_INFERER_FACTORY_H
