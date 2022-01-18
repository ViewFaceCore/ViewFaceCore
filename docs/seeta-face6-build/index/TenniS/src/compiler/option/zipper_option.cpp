//
// Created by kier on 19-5-7.
//

#include <compiler/option/zipper_option.h>
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"

#include <valarray>

namespace ts {
    static std::vector<const ZipperOption*> &GetStaticOptions() {
        static std::vector<const ZipperOption*> options;
        return options;
    }

    const std::vector<const ZipperOption *> &GetFullOptions() {
        return GetStaticOptions();
    }

    void RegisterOption(const ZipperOption *option) {
        auto &options = GetStaticOptions();
        options.emplace_back(option);
    }

    class EmptyZipperOption : public ZipperOption {
    public:
        bool zip(const ComputingDevice &device, Node node, Node &zipped_node) const final {
            return false;
        }
    };
}

//TS_REGISTER_ZIPPER_OPTION(ts::EmptyZipperOption)
