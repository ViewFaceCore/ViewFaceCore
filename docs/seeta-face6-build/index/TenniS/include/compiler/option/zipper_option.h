//
// Created by kier on 19-5-7.
//

#ifndef TENSORSTACK_COMPILER_OPTION_ZIPPER_OPTION_H
#define TENSORSTACK_COMPILER_OPTION_ZIPPER_OPTION_H

#include "module/graph.h"
#include "utils/static.h"

namespace ts {

    class ZipperOption {
    public:
        using self = ZipperOption;

        virtual ~ZipperOption() = default;

        /**
         * zip node on device
         * @param device computing device
         * @param node checking node
         * @param zipped_node get zipped_node if return true
         * @return zipped return true, or false
         */
        virtual bool zip(const ComputingDevice &device, Node node, Node &zipped_node) const = 0;
    };

    const std::vector<const ZipperOption*> &GetFullOptions();

    void RegisterOption(const ZipperOption *option);
}

#define TS_REGISTER_ZIPPER_OPTION(option) \
namespace { \
    static option ts_serial_name(_ts_zipper_option); \
    TS_STATIC_ACTION(ts::RegisterOption, &(ts_serial_name(_ts_zipper_option))); \
}

#endif //TENSORSTACK_COMPILER_OPTION_ZIPPER_OPTION_H
