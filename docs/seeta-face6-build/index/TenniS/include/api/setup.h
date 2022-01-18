//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_SETUP_H
#define TENNIS_API_SETUP_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @return ts_true if succeed.
 * @note Happen nothing if failed.
 */
TENNIS_C_API ts_bool ts_setup();

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_SETUP_H
