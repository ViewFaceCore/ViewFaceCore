//
// Created by kier on 2018/11/1.
//

#ifndef TENSORSTACK_COMPILER_ZIPPER_H
#define TENSORSTACK_COMPILER_ZIPPER_H


#include <core/device.h>
#include "module/graph.h"

namespace ts {
    class ZipperOption;
    /**
     * zip TGraph to ZGraph
     * may remove or add nodes
     */
    class Zipper {
    public:
        using self = Zipper;

        explicit Zipper(const ComputingDevice &device);

        explicit Zipper(const ComputingDevice &device, const std::string &params);

        ~Zipper();

        std::vector<Node> zip(const std::vector<Node> &nodes) const;

    private:
        ComputingDevice m_device;

        std::vector<const ZipperOption *> m_options;
    };
}


#endif //TENSORSTACK_COMPILER_ZIPPER_H
