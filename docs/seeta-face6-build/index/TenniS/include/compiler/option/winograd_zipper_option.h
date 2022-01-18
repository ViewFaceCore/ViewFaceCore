//
// Created by kier on 2019-06-10.
//

#ifndef TENSORSTACK_WINOGRAD_ZIPPER_OPTION_H
#define TENSORSTACK_WINOGRAD_ZIPPER_OPTION_H

#include "zipper_option.h"


namespace ts {
    class Conv2dZipperOption : public ZipperOption {
    public:
        bool zip(const ComputingDevice &device, Node node, Node &zipped_node) const final;
    };
}


#endif //TENSORSTACK_WINOGRAD_ZIPPER_OPTION_H
