//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_MODULE_H
#define TENNIS_API_MODULE_H

#include "common.h"
#include "stream.h"
#include "device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Module contains struct and input, output of networks.
 */
struct ts_Module;
typedef struct ts_Module ts_Module;

/**
 * Serialized module format
 */
enum ts_SerializationFormat {
    TS_BINARY   = 0,    // BINARY file format
    TS_TEXT     = 1,    // TEXT file format
};
typedef enum ts_SerializationFormat ts_SerializationFormat;

// Module's API

/**
 * Load module from given filename.
 * @param filename
 * @param format @sa ts_SerializationFormat, only support TS_BINARY in this version.
 * @return New reference. Return NULL if failed.
 * @note call @see ts_free_Module to free ts_Module
 */
TENNIS_C_API ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format);

/**
 * Load module from given stream.
 * @param obj object pointer pass to reader
 * @param reader stream reader
 * @param format @sa ts_SerializationFormat, only support TS_BINARY in this version.
 * @return New reference. Return NULL if failed.
 * @note call @see ts_free_Module to free ts_Module
 */
TENNIS_C_API ts_Module *ts_Module_LoadFromStream(void *obj, ts_stream_read *reader, ts_SerializationFormat format);

/**
 * Free module.
 * @param module the return value of ts_Module_Load<XXX>
 * Happen nothing if failed.
 */
TENNIS_C_API void ts_free_Module(const ts_Module *module);

/**
 * Fusion two modules to one.
 * @param in input module
 * @param in_out_slot input module's output slot
 * @param out output module
 * @param in_out_slot output module's input slot
 * @return New reference. Return NULL if failed.
 * @note call @see ts_free_Module to free ts_Module
 * @note new module's input and output is sorted all `in` and `out` modules left input and output nodes
 */
TENNIS_C_API ts_Module *ts_Module_Fusion(const ts_Module *in, int32_t in_out_slot, const ts_Module *out, int32_t out_in_slot);


#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_MODULE_H
