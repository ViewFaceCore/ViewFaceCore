//
// Created by kier on 19-5-8.
//

#ifndef TENNIS_API_OPERATOR_H
#define TENNIS_API_OPERATOR_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Operator's param
 */
struct ts_OperatorParams;
typedef struct ts_OperatorParams ts_OperatorParams;

/**
 * Operator's context, which can get context running.
 * @sa ts_OperatorContext_cuda_stream and @sa TS_CUDA_STREAM
 */
struct ts_OperatorContext;
typedef struct ts_OperatorContext ts_OperatorContext;

/**
 * Get param value from ts_OperatorParams by name
 * @param dict instance of operator's param
 * @param param param name
 * @return new reference, NULL if param is not exist
 */
TENNIS_C_API ts_Tensor *ts_OperatorParams_get(const ts_OperatorParams *dict, const char *param);

/**
 *
 * @return new object of operator
 */
typedef void *ts_new_Operator();

/**
 *
 * @param op return value of @see ts_new_Operator
 */
typedef void ts_free_Operator(const void *op);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param dict params dict
 */
typedef void ts_Operator_init(void *op, const ts_OperatorParams *dict, ts_OperatorContext *context);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param dict params dict
 * @return return ts_false if init failed
 */
typedef ts_bool ts_Operator_init_ex(void *op, const ts_OperatorParams *dict, ts_OperatorContext *context);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @return shape of ready values
 * infer return format int array tell the packed tensor shape
 * first element is fields_count, follow are each filed proto
 * for example, [2, TS_FLOAT32, 2, 4, 3, TS_INT32, 3, 5, 6, 7] means:
 *              {float32:[4, 3], int32:[5, 6, 7]}
 */
typedef ts_Tensor *ts_Operator_infer(void *op, int32_t argc, ts_Tensor **argv, ts_OperatorContext *context);

/**
 *
 * @param op return value of @see ts_new_Operator
 * @param argc argument count
 * @param argv argument values, are borrowed refs
 * @return packed values
 * @note return ts_new_Tensor_in_flow value
 */
typedef ts_Tensor *ts_Operator_run(void *op, int32_t argc, ts_Tensor **argv, ts_OperatorContext *context);


/**
 * @param device register device
 * @param op register op
 * @param f_new function to new Operator, will parsed as f_new(arg)
 * @param f_free function to free parameter
 * @param f_init function to init operator with given attributes
 * @param f_infer function to infer data shape
 * @param f_run function to run operator
 */
TENNIS_C_API void ts_Operator_Register(
        const char *device, const char *op,
        ts_new_Operator *f_new,
        ts_free_Operator *f_free,
        ts_Operator_init *f_init,
        ts_Operator_infer *f_infer,
        ts_Operator_run *f_run);


/**
 * @param device register device
 * @param op register op
 * @param f_new function to new Operator, will parsed as f_new(arg)
 * @param f_free function to free parameter
 * @param f_init function to init operator with given attributes
 * @param f_infer function to infer data shape
 * @param f_run function to run operator
 */
TENNIS_C_API void ts_Operator_RegisterEx(
        const char *device, const char *op,
        ts_new_Operator *f_new,
        ts_free_Operator *f_free,
        ts_Operator_init_ex *f_init,
        ts_Operator_infer *f_infer,
        ts_Operator_run *f_run);

/**
 * Throw exception message to break running in operator's functions
 * @param message throw message
 */
TENNIS_C_API void ts_Operator_Throw(const char *message);

/**
 * Throw exception message to break running in operator's functions
 * @param message throw message
 * @param filename Exceptional filename
 * @param line_number Exceptional line number
 */
TENNIS_C_API void ts_Operator_ThrowV2(const char *message, const char *filename, int32_t line_number);

/**
 * Throw exception message to break running in operator's functions
 * @param message throw message
 * @note auto gather filename and line number information.
 */
#define TS_C_THROW(message) \
    ts_Operator_ThrowV2((message), __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_OPERATOR_H
