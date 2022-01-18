//
// Created by yang on 2020/2/21.
//

#ifndef TENNIS_API_API_H
#define TENNIS_API_API_H

#include "api/common.h"
#include "api/device.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ts_op_creator_map;
typedef struct ts_op_creator_map ts_op_creator_map;

struct ts_device_context;
typedef struct ts_device_context ts_device_context;

TENNIS_C_API ts_op_creator_map* ts_plugin_get_creator_map();

TENNIS_C_API void ts_plugin_flush_creator(ts_op_creator_map* creator_map);

TENNIS_C_API void ts_plugin_free_creator_map(ts_op_creator_map* creator_map);

TENNIS_C_API ts_device_context* ts_plugin_initial_device_context(const ts_Device *device);

TENNIS_C_API void ts_plugin_free_device_context(ts_device_context* device);

TENNIS_C_API void ts_plugin_bind_device_context(ts_device_context* device);

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_API_H
