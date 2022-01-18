//
// Created by kier on 2019/3/16.
//

#ifndef TENSORSTACK_BOARD_DUMP_H
#define TENSORSTACK_BOARD_DUMP_H

#include "runtime/workbench.h"


namespace ts {
    TS_DEBUG_API std::vector<Tensor> track(const std::string &dump_root, Workbench::shared bench, const std::vector<Tensor> &input);
}


#endif //TENSORSTACK_BOARD_DUMP_H
