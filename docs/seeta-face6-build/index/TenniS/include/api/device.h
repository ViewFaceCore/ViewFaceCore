//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_DEVICE_H
#define TENNIS_API_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * Device description
 */
struct ts_Device {
    const char *type;   ///< device type, like cpu or gpu
    int32_t id;         ///< device id
};
typedef struct ts_Device ts_Device;

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_DEVICE_H
