//
// Created by kier on 2019/3/16.
//

#include "api/common.h"
#include "errno.h"

using namespace ts;
using namespace api;

const char *ts_last_error_message() {
    return GetLEM().c_str();
}

void ts_set_error_message(const char *message) {
    SetLEM(message);
}
