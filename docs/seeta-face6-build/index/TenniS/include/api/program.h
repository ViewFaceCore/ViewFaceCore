//
// Created by kier on 19-5-7.
//

#ifndef TENNIS_API_PROGORAM_H
#define TENNIS_API_PROGORAM_H

#include "device.h"
#include "common.h"
#include "tensor.h"
#include "module.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Program contain compiled network, ready to run in given Workbench
 */
struct ts_Program;
typedef struct ts_Program ts_Program;

// Workbench API

/**
 * Compile module to program running on device
 * @param module ready module
 * @param device @sa ts_Device
 * @return new reference program, NULL if failed.
 * @note Call ts_Workbench_setup_context before compile, or call ts_Workbench_compile instead
 * @note call @see ts_free_Program to free ts_Program
 */
TENNIS_C_API ts_Program *ts_Program_Compile(const ts_Module *module, const ts_Device *device);

/**
 * Free program.
 * @param module the return value of ts_Program_Compile or ts_Workbench_compile
 * Happen nothing if failed.
 */
TENNIS_C_API void ts_free_Program(const ts_Program *program);

/**
 * Clone existing program.
 * @param module instance of program
 * @return new reference, NULL if failed
 * Each thread must has thead-local program and workbench
 */
TENNIS_C_API ts_Program *ts_Program_clone(ts_Program *program);

/**
 * Get input number of program
 * @param module instance of program
 * @return input count
 */
TENNIS_C_API int32_t ts_Program_input_count(ts_Program *program);

/**
 * Get output number of program
 * @param module instance of program
 * @return output count
 */
TENNIS_C_API int32_t ts_Program_output_count(ts_Program *program);

/**
 * Compile module to program running on device
 * @param module ready module
 * @param device @sa ts_Device
 * @param options compile option, split with space word
 * @return new reference program, NULL if failed.
 * @note Call ts_Workbench_setup_context before compile, or call ts_Workbench_compile instead
 * @note call @see ts_free_Program to free ts_Program
 * Option can have:
 * 1. "--float16" using float16 operator
 * 2. "--winograd" using winograd conv2d
 * Reservation options:
 * 1. "--pack" Default ON, pack weights
 * 2. "--filter" Default OFF, filter const values, set values to zero which smaller than FLT_EPSILON
 */
TENNIS_C_API ts_Program *ts_Program_Compile_v2(const ts_Module *module, const ts_Device *device,
                                               const char *options);
/**
 * Set operator's param value.
 * @param program instance of program
 * @param node_name node name
 * @param param param name
 * @param value param value
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_Program_set_operator_param(ts_Program *program, const char *node_name,
                                                   const char *param, const ts_Tensor *value);


#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_PROGORAM_H
