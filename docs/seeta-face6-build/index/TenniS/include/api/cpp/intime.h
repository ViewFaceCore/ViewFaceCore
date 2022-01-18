//
// Created by kier on 19-5-13.
//

#ifndef TENNIS_API_CPP_INTIME_H
#define TENNIS_API_CPP_INTIME_H

#include "tensor.h"
#include "../intime.h"
#include "../image_filter.h"

#include <array>

namespace ts {
    namespace api {
        namespace intime {
            struct DimPadding {
                DimPadding() = default;

                DimPadding(int32_t first, int32_t second)
                        : first(first), second(second) {}

                int32_t first = 0;
                int32_t second = 0;
            };

            /**
             * @see ts_ResizeMethod
             */
            enum class ResizeMethod : int32_t {
                BICUBIC = TS_RESIZE_BICUBIC,
                BILINEAR = TS_RESIZE_BILINEAR,
                NEAREST = TS_RESIZE_NEAREST,
            };

            inline Tensor transpose(const Tensor &x, const std::vector<int32_t> &permute) {
                auto y = ts_intime_transpose(x.get_raw(), permute.data(), int32_t(permute.size()));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor sigmoid(const Tensor &x) {
                auto y = ts_intime_sigmoid(x.get_raw());
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor gather(const Tensor &x, const Tensor &indices, int32_t axis) {
                auto y = ts_intime_gather(x.get_raw(), indices.get_raw(), axis);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor concat(const std::vector<Tensor> &x, int32_t dim) {
                std::vector<ts_Tensor *> inputs;
                for (auto &input : x) {
                    inputs.emplace_back(input.get_raw());
                }
                auto y = ts_intime_concat(inputs.data(), int32_t(x.size()), dim);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor softmax(const Tensor &x, int32_t dim, bool smooth = true) {
                auto y = ts_intime_softmax(x.get_raw(), dim, ts_bool(smooth));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor pad(const Tensor &x, const Tensor &padding, float padding_value = 0) {
                auto y = ts_intime_pad(x.get_raw(), padding.get_raw(), padding_value);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor pad(const Tensor &x, const std::vector<DimPadding> &padding, float padding_value = 0) {
                auto padding_tensor = tensor::build(INT32, {int(padding.size()), 2}, &padding[0].first);
                return pad(x, padding_tensor, padding_value);
            }

            inline Tensor cast(const Tensor &x, DTYPE dtype) {
                auto y = ts_intime_cast(x.get_raw(), dtype);
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor resize2d(const Tensor &x, const Tensor &size, ResizeMethod method = ResizeMethod::BILINEAR) {
                auto y = ts_intime_resize2d(x.get_raw(), size.get_raw(), int32_t(method));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor affine_sample2d(
                    const Tensor &x,
                    const Tensor &size,
                    const Tensor &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                auto y = ts_intime_affine_sample2d(
                        x.get_raw(),
                        size.get_raw(),
                        affine.get_raw(),
                        dim,
                        outer_value,
                        int32_t(method));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor affine_sample2d(
                    const Tensor &x,
                    const std::array<int32_t, 2> &size,
                    const Tensor &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_sample2d(
                        x,
                        tensor::build(INT32, Shape({2,}), &size[0]),
                        affine,
                        dim, outer_value, method);
            }

            inline Tensor affine_sample2d(
                    const Tensor &x,
                    const Tensor &size,
                    const std::array<float, 9> &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_sample2d(
                        x,
                        size,
                        tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                        dim, outer_value, method);
            }

            inline Tensor affine_sample2d(
                    const Tensor &x,
                    const std::array<int32_t, 2> &size,
                    const std::array<float, 9> &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_sample2d(
                        x,
                        tensor::build(INT32, Shape({2,}), &size[0]),
                        tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                        dim, outer_value, method);
            }

            inline Tensor affine_on_sample2d(
                    const Tensor &x,
                    const Tensor &size,
                    const Tensor &affine,
                    int32_t dim = -2,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                auto y = ts_intime_affine_on_sample2d(
                        x.get_raw(),
                        size.get_raw(),
                        affine.get_raw(),
                        dim,
                        int32_t(method));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }

            inline Tensor affine_on_sample2d(
                    const Tensor &x,
                    const std::array<int32_t, 2> &size,
                    const Tensor &affine,
                    int32_t dim = -2,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_on_sample2d(
                        x,
                        tensor::build(INT32, Shape({2,}), &size[0]),
                        affine,
                        dim, method);
            }

            inline Tensor affine_on_sample2d(
                    const Tensor &x,
                    const Tensor &size,
                    const std::array<float, 9> &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_on_sample2d(
                        x,
                        size,
                        tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                        dim, method);
            }

            inline Tensor affine_on_sample2d(
                    const Tensor &x,
                    const std::array<int32_t, 2> &size,
                    const std::array<float, 9> &affine,
                    int32_t dim = -2,
                    float outer_value = 0,
                    ResizeMethod method = ResizeMethod::BILINEAR) {
                return affine_on_sample2d(
                        x,
                        tensor::build(INT32, Shape({2,}), &size[0]),
                        tensor::build(FLOAT32, Shape({3, 3}), &affine[0]),
                        dim, method);
            }

            inline int64_t memcpy(
                    Tensor &dst_desc, int64_t dst_shift,
                    const Tensor &src_desc, int64_t src_shift,
                    int64_t size) {
                return ts_intime_memcpy(
                        dst_desc.get_raw(), nullptr, dst_shift,
                        src_desc.get_raw(), nullptr, src_shift,
                        size);
            }

            inline int64_t memcpy(
                    Tensor &dst_desc, void *dst_data, int64_t dst_shift,
                    const Tensor &src_desc, const void *src_data, int64_t src_shift,
                    int64_t size) {
                return ts_intime_memcpy(
                        dst_desc.get_raw(), dst_data, dst_shift,
                        src_desc.get_raw(), src_data, src_shift,
                        size);
            }

            inline Tensor matmul(const Tensor &A, const Tensor &B, bool transpose = false) {
                auto y = ts_intime_matmul(A.get_raw(), B.get_raw(), ts_bool(transpose));
                TS_API_AUTO_CHECK(y != nullptr);
                return Tensor::NewRef(y);
            }
        }
    }
}

#endif //TENNIS_API_CPP_INTIME_H
