//
// Created by kier on 2019/3/16.
//

#include "api/setup.h"
#include "declaration.h"

#include "global/setup.h"

using namespace ts;

ts_bool ts_setup() {
    TRY_HEAD
    setup();
    RETURN_OR_CATCH(ts_true, ts_false);
}

