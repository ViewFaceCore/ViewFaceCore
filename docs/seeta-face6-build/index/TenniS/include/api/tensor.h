//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_TENSOR_H
#define TENNIS_API_TENSOR_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * class of NDTensor
 */
struct ts_Tensor;
typedef struct ts_Tensor ts_Tensor;

/**
 * tensor value data type
 */
enum ts_DTYPE {
    TS_VOID        = 0,
    TS_INT8        = 1,
    TS_UINT8       = 2,
    TS_INT16       = 3,
    TS_UINT16      = 4,
    TS_INT32       = 5,
    TS_UINT32      = 6,
    TS_INT64       = 7,
    TS_UINT64      = 8,
    TS_FLOAT32     = 10,
    TS_FLOAT64     = 11,
    TS_CHAR8       = 13,
};
typedef enum ts_DTYPE ts_DTYPE;

/**
 * Flow memory description
 */
enum ts_InFlow {
    TS_HOST     = 0,    ///< Memory is on host(CPU)
    TS_DEVICE   = 1,    ///< Memory is on device, described in ts_Workbench
};
typedef enum ts_InFlow ts_InFlow;

// Tensor's API

/**
 * New tensor
 * @param shape shape of new tensor
 * @param shape_len length of given shape
 * @param dtype data type of tensor
 * @param data initialization data on cpu, must have same data type as dtype. passing NULL for no initialization.
 * @return new reference, NULL if failed.
 * @note @sa ts_free_Tensor to free ts_Tensor
 */
TENNIS_C_API ts_Tensor *ts_new_Tensor(const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Free tensor.
 * @param tensor the instance of tensor
 * Happen nothing if failed.
 */
TENNIS_C_API void ts_free_Tensor(const ts_Tensor *tensor);

/**
 * Return tensor's shape
 * @param tensor the instance of tensor
 * @return pointer to shape, borrowed pointer.
 * @note shape length can be get by @sa ts_Tensor_shape_size
 */
TENNIS_C_API const int32_t *ts_Tensor_shape(ts_Tensor *tensor);

/**
 * Return tensor's shape size
 * @param tensor the instance of tensor
 * @return pointer to shape size, zero if failed.
 */
TENNIS_C_API int32_t ts_Tensor_shape_size(ts_Tensor *tensor);

/**
 * Return tensor's data type
 * @param tensor the instance of tensor
 * @return data type, TS_VOID if failed.
 */
TENNIS_C_API ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor);

/**
 * Return tensor's data
 * @param tensor the instance of tensor
 * @return pointer to data, borrowed pointer
 */
TENNIS_C_API void* ts_Tensor_data(ts_Tensor *tensor);

/**
 * Clone tensor on original device.
 * @param tensor the instance of tensor
 * @return new reference. NULL if failed.
 */
TENNIS_C_API ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor);

/**
 * Sync tensor's data to cpu
 * @param tensor the instance of tensor
 * @return false if failed.
 */
TENNIS_C_API ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor);

/**
 * Cast tensor to given dtype. new tensor data would force on CPU
 * @param tensor the instance of tensor
 * @param dtype wanted data type
 * @return new reference, NULL if failed.
 * @note new tensor data would force on CPU
 */
TENNIS_C_API ts_Tensor *ts_Tensor_cast(ts_Tensor *tensor, ts_DTYPE dtype);

/**
 * reshape tensor to given shape. data would not change.
 * @param tensor the instance of tensor
 * @param shape new shape
 * @param shape_len length of shape
 * @return new reference, NULL if failed.
 * @note the prod new shape must equal to old shape
 */
TENNIS_C_API ts_Tensor *ts_Tensor_reshape(ts_Tensor *tensor, const int32_t *shape, int32_t shape_len);

/**
 * New tensor
 * @param in_flow tell where tensor's data store
 * @param shape shape of new tensor
 * @param shape_len length of given shape
 * @param dtype data type of tensor
 * @param data initialization data on cpu, must have same data type as dtype. passing NULL for no initialization.
 * @return new reference, NULL if failed.
 * @note @sa ts_free_Tensor to free ts_Tensor
 * @note call ts_Workbench_setup_context to fix exception "Empty context:<ts::Workbench>"
 */
TENNIS_C_API ts_Tensor *ts_new_Tensor_in_flow(ts_InFlow in_flow, const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data);

/**
 * Get tensor view on device in flow.
 * @param tensor the instance of tensor
 * @param in_flow tell where tensor's data store
 * @return new reference, NULL if failed.
 * @note call ts_Workbench_setup_context to fix exception "Empty context:<ts::Workbench>"
 */
TENNIS_C_API ts_Tensor *ts_Tensor_view_in_flow(ts_Tensor *tensor, ts_InFlow in_flow);

/**
 * Get tensor field in packed tensor
 * @param tensor the instance of tensor
 * @param index field index
 * @return new reference, NULL if failed
 * if index < 0 or index >= fields_count, API will return NULL
 */
TENNIS_C_API ts_Tensor *ts_Tensor_field(ts_Tensor *tensor, int32_t index);

/**
 * Return false if failed. False also mean no packed
 * @param tensor the instance of tensor
 * @return if packed
 */
TENNIS_C_API ts_bool ts_Tensor_packed(ts_Tensor *tensor);

/**
 * Get fields count
 * @param tensor the instance of tensor
 * @return fields count, 0 if failed.
 */
TENNIS_C_API int32_t ts_Tensor_fields_count(ts_Tensor *tensor);

/**
 * Pack fields to new tensor.
 * @param fields list of tensor
 * @param count length of fields
 * @return new reference, NULL if failed.
 */
TENNIS_C_API ts_Tensor *ts_Tensor_pack(ts_Tensor **fields, int32_t count);

/**
 * Return slice tensor, slice tensor will invalid after origin tensor free
 * @param tensor the instance of tensor
 * @param i tell indices
 * @return tensor[i, :]
 */
TENNIS_C_API ts_Tensor *ts_Tensor_slice(ts_Tensor *tensor, int32_t i);

/**
 * Return slice tensor, slice tensor will invalid after origin tensor free
 * @param tensor the instance of tensor
 * @param beg tell indices
 * @param end tell indices
 * @return tensor[beg:end, :]
 */
TENNIS_C_API ts_Tensor *ts_Tensor_slice_v2(ts_Tensor *tensor, int32_t beg, int32_t end);

/**
 * Save tensor to file
 * @param path file path
 * @param tensor the instance of tensor
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_Tensor_save(const char *path, const ts_Tensor *tensor);

/**
 * Load tensor from file
 * @param path file path
 * @return NULL if failed
 */
TENNIS_C_API ts_Tensor *ts_Tensor_load(const char *path);


#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_TENSOR_H
