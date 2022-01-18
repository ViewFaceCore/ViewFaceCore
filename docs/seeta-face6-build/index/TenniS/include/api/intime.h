//
// Created by kier on 19-5-13.
//

#ifndef TENNIS_API_INTIME_H
#define TENNIS_API_INTIME_H

#include "common.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return transposed x with permute. `out = transpose(x, axes=permute)`.
 * @param x input tensor
 * @param permute dest tensor axes
 * @param len length of permute
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENNIS_C_API ts_Tensor *ts_intime_transpose(const ts_Tensor *x, const int32_t *permute, int32_t len);

/**
 * Return sigmoid x. `out = 1 / (1 + e^{-x})`
 * @param x input tensor
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENNIS_C_API ts_Tensor *ts_intime_sigmoid(const ts_Tensor *x);

/**
 * Return gathered x. `out = take(x, indices=indices, axis=axis)`
 * @param x input tensor
 * @param indices indices
 * @param axis axis
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENNIS_C_API ts_Tensor *ts_intime_gather(const ts_Tensor *x, const ts_Tensor *indices, int32_t axis);

/**
 * Return concat x. `out = concat(x, axis=dim)`
 * @param x input tensors
 * @param len length of input tensors
 * @param dim concat dim
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENNIS_C_API ts_Tensor *ts_intime_concat(const ts_Tensor *const *x, int32_t len, int32_t dim);

/**
 * Return softmax x. output y.
 * @param x input tensor
 * @param dim softmax on given dim
 * @param smooth if smooth mode
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 * if not smooth:
 * ```
 * y_i = exp(x_i) / \sum{exp(x_i)}
 * ```
 * else:
 * ```
 * t_i = x_i - max(x)
 * y_i = exp(t_i) / \sum{exp(t_i)}
 * in framework like caffe, smooth is true.
 */
TENNIS_C_API ts_Tensor *ts_intime_softmax(const ts_Tensor *x, int32_t dim, ts_bool smooth);

/**
 * Return pad x.
 * @param x input tensor
 * @param padding Int[_, 2] than first axis must equal to dims of x
 * @param padding_value padding value
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 * @note padding size can be neg value.
 */
TENNIS_C_API ts_Tensor *ts_intime_pad(const ts_Tensor *x, const ts_Tensor *padding, float padding_value);

/**
 * Return given dtype tensor.
 * @param x input tensor
 * @param dtype @sa ts_DTYPE
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 *
 */
TENNIS_C_API ts_Tensor *ts_intime_cast(const ts_Tensor *x, ts_DTYPE dtype);

/**
 * Return resized tensor
 * @param x input tensor
 * @param size size tensor, size.size(0) same as x.dims
 * @param method 0-BILINEAR, 1-BICUBIC, 2-NEAREST, @sa ts_ResizeMethod
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 */
TENNIS_C_API ts_Tensor *ts_intime_resize2d(const ts_Tensor *x, const ts_Tensor *size, int32_t method);

/**
 * Return sample2d tensor, out_position = affine * in_position
 * @param x input tensor
 * @param size size tensor like Int[2], means {height, width}
 * @param affine affine tensor like Float[3, 3]
 * @param method 0-BILINEAR, 1-BICUBIC, 2-NEAREST, @sa ts_ResizeMethod
 * @param dim first dim of image height and width
 * @param outer_value set value to outer_value if sample out of image
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 */
TENNIS_C_API ts_Tensor *ts_intime_affine_sample2d(
        const ts_Tensor *x,
        const ts_Tensor *size,
        const ts_Tensor *affine,
        int32_t dim, float outer_value, int32_t method);

/**
 * Return sample2d tensor, out_position = affine * in_position
 * @param x input tensor
 * @param size size tensor like Int[2], means {height, width}
 * @param affine affine tensor like Float[3, 3]
 * @param dim first dim of image height and width
 * @param method 0-BILINEAR, 1-BICUBIC, 2-NEAREST, @sa ts_ResizeMethod
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note ON = Outer value is Nearest pixel
 */
TENNIS_C_API ts_Tensor *ts_intime_affine_on_sample2d(
        const ts_Tensor *x,
        const ts_Tensor *size,
        const ts_Tensor *affine,
        int32_t dim, int32_t method);

/**
 * Do memcpy from different tensor
 * @param dst_desc tensor description, tell API device information
 * @param dst_ptr tensor data pointer, must match description. it would be dst_desc.data() if pointer is nullptr
 * @param dst_shift memory will copy to dst_ptr + dst_shift
 * @param src_desc tensor description, tell API device information
 * @param src_ptr tensor data pointer, must match description. it would be dst_desc.data() if pointer is nullptr
 * @param src_shift memory will copy from src_ptr + src_shift
 * @param size the sizeof memory would copy
 * @return copied memory size
 */
TENNIS_C_API int64_t ts_intime_memcpy(
        ts_Tensor *dst_desc, void *dst_ptr, int64_t dst_shift,
        const ts_Tensor *src_desc, const void *src_ptr, int64_t src_shift,
        int64_t size);

/**
 * Return A * B. or A * B^T if transpose
 * @param A lhs tensor
 * @param B rhs tensor
 * @param transpose if B transposed
 * @return new reference tensor, nullptr if failed.
 * @note call ts_Workbench_setup_context to fix Exception "Must bind Workbench before run"
 * @note output tensor should on device setting in setup context ts_Workbench
 */
TENNIS_C_API ts_Tensor *ts_intime_matmul(const ts_Tensor *A, const ts_Tensor *B, ts_bool transpose);


#ifdef __cplusplus
}
#endif

#endif //TENSORSTACK_INTIME_H
