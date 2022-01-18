//
// Created by kier on 2019/11/20.
//

#include "global/shape_inferer_factory.h"

#include "utils/static.h"

#include <map>
#include <cstdlib>
#include <iostream>
#include <algorithm>

namespace ts {
    static std::map<std::string, ShapeInferer::function> &MapOpInferer() {
        static std::map<std::string, ShapeInferer::function> map_op_inferer;
        return map_op_inferer;
    };

    ShapeInferer::function ShapeInferer::Query(const std::string &op) TS_NOEXCEPT {
        auto &map_op_inferer = MapOpInferer();
        auto op_inferer = map_op_inferer.find(op);
        if (op_inferer != map_op_inferer.end()) {
            return op_inferer->second;
        }
        return ShapeInferer::function(nullptr);
    }

    void ShapeInferer::Register(const std::string &op, const function &inferer) TS_NOEXCEPT {
        auto &map_op_inferer = MapOpInferer();
        map_op_inferer[op] = inferer;
    }

    void ShapeInferer::Clear() {
        auto &map_op_inferer = MapOpInferer();
        map_op_inferer.clear();
    }

    std::set<std::string> ShapeInferer::AllKeys() TS_NOEXCEPT {
        auto &map_op_inferer = MapOpInferer();
        std::set<std::string> set_op;
        for (auto &op : map_op_inferer) {
            set_op.insert(op.first);
        }
        return set_op;
    }
}

