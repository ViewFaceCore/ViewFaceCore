//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_WORKBENCH_H
#define TENNIS_API_WORKBENCH_H

#include "common.h"
#include "tensor.h"
#include "module.h"
#include "image_filter.h"
#include "program.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run program on ME!
 */
struct ts_Workbench;
typedef struct ts_Workbench ts_Workbench;

/**
 * CPU information, only support since v0.3.2
 */
enum ts_CpuPowerMode {
    TS_CPU_BALANCE = 0,     ///< balance using big and little cores
    TS_CPU_BIG_CORE = 1,    ///< only used in big core
    TS_CPU_LITTLE_CORE = 2, ///< only used in little core
};

// Workbench API

/**
 * New workbench and setup compiled program with module.
 * @param module instance of module
 * @param device workbench computing device
 * @return new reference, NULL if failed.
 * @note @sa ts_free_Workbench to free ts_Workbench
 */
TENNIS_C_API ts_Workbench *ts_Workbench_Load(const ts_Module *module, const ts_Device *device);

/**
 * Free workbench.
 * @param workbench instance of workbench
 * Happen nothing if failed.
 */
TENNIS_C_API void ts_free_Workbench(const ts_Workbench *workbench);

/**
 * Clone existing ts_Workbench with setup program.
 * @param workbench instance of workbench
 * @return new reference, NULL if failed.
 */
TENNIS_C_API ts_Workbench *ts_Workbench_clone(ts_Workbench *workbench);

/**
 * Input i-th tensor.
 * @param workbench instance of workbench
 * @param i slot index
 * @param tensor input tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_input(ts_Workbench *workbench, int32_t i, const ts_Tensor *tensor);

/**
 * Input named tensor.
 * @param workbench instance of workbench
 * @param name slot name
 * @param tensor input tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_input_by_name(ts_Workbench *workbench, const char *name, const ts_Tensor *tensor);

/**
 * Run network. then get output by ts_Workbench_output or ts_Workbench_output_by_name
 * @param workbench instance of workbench
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_run(ts_Workbench *workbench);

/**
 * Get output i-th tensor.
 * @param workbench instance of workbench
 * @param i slot index
 * @param tensor output tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_output(ts_Workbench *workbench, int32_t i, ts_Tensor *tensor);

/**
 * Get output named tensor.
 * @param workbench instance of workbench
 * @param name slot name
 * @param tensor output tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_output_by_name(ts_Workbench *workbench, const char *name, ts_Tensor *tensor);

/**
 * Set computing thread number
 * @param workbench instance of workbench
 * @param number thread number
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_set_computing_thread_number(ts_Workbench *workbench, int32_t number);

/**
 * Bind filter on i-th input.
 * @param workbench instance of workbench
 * @param i slot index
 * @param filter output tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, const ts_ImageFilter *filter);

/**
 * Bind filter on named input.
 * @param workbench instance of workbench
 * @param name slot name
 * @param filter output tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, const ts_ImageFilter *filter);

/**
 * New workbench.
 * @param device @sa ts_Device
 * @return new reference, NULL if failed.
 */
TENNIS_C_API ts_Workbench *ts_new_Workbench(const ts_Device *device);

/**
 * Setup program on workbench. workbench device must same with program device.
 * @param workbench instance of workbench
 * @param program instance of program
 * @return false if failed.
 * @note this api will change next input, output and run APIs
 */
TENNIS_C_API ts_bool ts_Workbench_setup(ts_Workbench *workbench, const ts_Program *program);

/**
 * Setup context of workbench.
 * @param workbench instance of workbench
 * @return false if failed.
 * @note this API influence:
 *     ts_new_Tensor_in_flow, ts_Program_Compile_XXX, ts_intime_XXX
 *     ts_Tensor_view_in_flow
 */
TENNIS_C_API ts_bool ts_Workbench_setup_context(ts_Workbench *workbench);

/**
 * Compile module to program
 * @param workbench instance of workbench
 * @param module instance of module
 * @return new reference, NULL if failed.
 */
TENNIS_C_API ts_Program *ts_Workbench_compile(ts_Workbench *workbench, const ts_Module *module);

/**
 * @deprecated Use ts_Workbench_setup_context instead
 */
TENNIS_C_API ts_bool ts_Workbench_setup_device(ts_Workbench *workbench);

/**
 * @deprecated Use ts_Workbench_setup_context instead
 */
TENNIS_C_API ts_bool ts_Workbench_setup_runtime(ts_Workbench *workbench);

/**
 * Get setup program's input count
 * @param workbench instance of workbench
 * @return input count
 */
TENNIS_C_API int32_t ts_Workbench_input_count(ts_Workbench *workbench);

/**
 * Get setup program's output count
 * @param workbench instance of workbench
 * @return output count
 */
TENNIS_C_API int32_t ts_Workbench_output_count(ts_Workbench *workbench);

/**
 * Run network hook inner node value, for debug using.
 * @param workbench instance of workbench
 * @param node_names hook node_names
 * @param len length of node_names
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Workbench_run_hook(ts_Workbench *workbench, const char **node_names, int32_t len);

/**
 * New workbench and setup compiled program with module.
 * @param module instance of module
 * @param device workbench computing device
 * @param options compile option, split with space word
 * @return new reference, NULL if failed.
 * @note @sa ts_free_Workbench to free ts_Workbench
 * Option can have:
 * 1. "--float16" using float16 operator
 * 2. "--winograd" using winograd conv2d
 * Reservation options:
 * 1. "--pack" Default ON, pack weights
 * 2. "--filter" Default OFF, filter const values, set values to zero which smaller than FLT_EPSILON
 */
TENNIS_C_API ts_Workbench *ts_Workbench_Load_v2(const ts_Module *module, const ts_Device *device,
                                                const char *options);

/**
 * Compile module to program
 * @param workbench instance of workbench
 * @param module instance of module
 * @param options compile option, split with space word
 * @return new reference, NULL if failed.
 * Option can have:
 * 1. "--float16" using float16 operator
 * 2. "--winograd" using winograd conv2d
 * Reservation options:
 * 1. "--pack" Default ON, pack weights
 * 2. "--filter" Default OFF, filter const values, set values to zero which smaller than FLT_EPSILON
 */
TENNIS_C_API ts_Program *ts_Workbench_compile_v2(ts_Workbench *workbench, const ts_Module *module,
                                                 const char *options);

/**
 * Set operator's param value.
 * @param workbench instance of workbench
 * @param node_name node name
 * @param param param name
 * @param value param value
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_Workbench_set_operator_param(ts_Workbench *workbench, const char *node_name,
                                                     const char *param, const ts_Tensor *value);

/**
 * Get operator's summary
 * @param workbench instance of workbench
 * @return summary string, NULL if failed
 * summary is a json string, like:
 *     {"device": "gpu:0", "thread": 4, "shared": "97.8MB", "memory": {"cpu:0": "32B", "gpu:0": "7.3MB"}}
 */
TENNIS_C_API const char *ts_Workbench_summary(ts_Workbench *workbench);

/**
 * Set workbench thread mode, effected now working thread
 * @param workbench instance of workbench
 * @param mode @ts_CPU_mode
 * @return false if failed
 * @note may call in each workbench in working thread.
 * @note this is testing API
 */
TENNIS_C_API ts_bool ts_Workbench_set_cpu_mode(ts_Workbench *workbench, ts_CpuPowerMode mode);


#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_WORKBENCH_H
