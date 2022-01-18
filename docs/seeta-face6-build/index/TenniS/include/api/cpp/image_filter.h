//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_CPP_IMAGE_FILTER_H
#define TENNIS_API_CPP_IMAGE_FILTER_H

#include "../image_filter.h"

#include "except.h"
#include "device.h"
#include "tensor.h"
#include "module.h"

#include <string>
#include <vector>

namespace ts {
    namespace api {
        /**
         * @see ts_ImageFilter
         */
        class ImageFilter {
        public:
            /**
             * @see ts_ResizeMethod
             */
            enum class ResizeMethod : int32_t {
                BICUBIC = TS_RESIZE_BICUBIC,
                BILINEAR = TS_RESIZE_BILINEAR,
                NEAREST = TS_RESIZE_NEAREST,
            };

            using self = ImageFilter;
            using raw = ts_ImageFilter;

            using shared = std::shared_ptr<self>;
            using shared_raw = std::shared_ptr<raw>;

            static self NewRef(raw *ptr) { return self(ptr); }

            ImageFilter(const self &) = default;

            ImageFilter &operator=(const self &) = default;

            raw *get_raw() const { return m_impl.get(); }

            bool operator==(std::nullptr_t) const { return get_raw() == nullptr; }

            bool operator!=(std::nullptr_t) const { return get_raw() != nullptr; }

            ImageFilter(std::nullptr_t) {}

            ImageFilter() : self(Device()) {}

            ImageFilter(const Device &device) : self(device.get_raw()) {}

            ImageFilter(const ts_Device *device) : self(ts_new_ImageFilter(device)) {
                TS_API_AUTO_CHECK(m_impl != nullptr);
            }

            void clear() {
                TS_API_AUTO_CHECK(ts_ImageFilter_clear(m_impl.get()));
            }

            void compile() {
                TS_API_AUTO_CHECK(ts_ImageFilter_compile(m_impl.get()));
            }

            void to_float() {
                TS_API_AUTO_CHECK(ts_ImageFilter_to_float(m_impl.get()));
            }

            void scale(float scale) {
                TS_API_AUTO_CHECK(ts_ImageFilter_scale(m_impl.get(), scale));
            }

            void sub_mean(const std::vector<float> &mean) {
                TS_API_AUTO_CHECK(ts_ImageFilter_sub_mean(m_impl.get(), mean.data(), int(mean.size())));
            }

            void div_std(const std::vector<float> &std) {
                TS_API_AUTO_CHECK(ts_ImageFilter_div_std(m_impl.get(), std.data(), int(std.size())));
            }

            void resize(int width, int height) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize(m_impl.get(), width, height));
            }

            void resize(int width) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize_scalar(m_impl.get(), width));
            }

            void center_crop(int width, int height) {
                TS_API_AUTO_CHECK(ts_ImageFilter_center_crop(m_impl.get(), width, height));
            }

            void center_crop(int width) {
                TS_API_AUTO_CHECK(ts_ImageFilter_center_crop(m_impl.get(), width, width));
            }

            void channel_swap(const std::vector<int> &shuffle) {
                TS_API_AUTO_CHECK(ts_ImageFilter_channel_swap(m_impl.get(), shuffle.data(), int(shuffle.size())));
            }

            void to_chw() {
                TS_API_AUTO_CHECK(ts_ImageFilter_to_chw(m_impl.get()));
            }

            void prewhiten() {
                TS_API_AUTO_CHECK(ts_ImageFilter_prewhiten(m_impl.get()));
            }

            void letterbox(int width, int height, float outer_value = 0) {
                TS_API_AUTO_CHECK(ts_ImageFilter_letterbox(m_impl.get(), width, height, outer_value));
            }

            void letterbox(int width, float outer_value = 0) {
                letterbox(width, width, outer_value);
            }

            void divide(int width, int height, float padding_value = 0) {
                TS_API_AUTO_CHECK(ts_ImageFilter_divided(m_impl.get(), width, height, padding_value));
            }

            void resize(int width, int height, ts_ResizeMethod method) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize_v2(m_impl.get(), width, height, method));
            }

            void resize(int width, ts_ResizeMethod method) {
                TS_API_AUTO_CHECK(ts_ImageFilter_resize_scalar_v2(m_impl.get(), width, method));
            }

            void letterbox(int width, int height, float outer_value, ts_ResizeMethod method) {
                TS_API_AUTO_CHECK(ts_ImageFilter_letterbox_v2(m_impl.get(), width, height, outer_value, method));
            }

            void resize(int width, int height, ResizeMethod method) {
                resize(width, height, ts_ResizeMethod(method));
            }

            void resize(int width, ResizeMethod method) {
                resize(width, ts_ResizeMethod(method));
            }

            void letterbox(int width, int height, float outer_value, ResizeMethod method) {
                letterbox(width, height, outer_value, ts_ResizeMethod(method));
            }

            void letterbox(int width, float outer_value, ResizeMethod method) {
                letterbox(width, width, outer_value, method);
            }

            void letterbox(int width, float outer_value, ts_ResizeMethod method) {
                letterbox(width, width, outer_value, method);
            }

            Tensor run(const Tensor &tensor) {
                return run(tensor.get_raw());
            }

            Tensor run(const ts_Tensor *tensor) {
                auto ret = ts_ImageFilter_run(m_impl.get(), tensor);
                TS_API_AUTO_CHECK(ret != nullptr);
                return Tensor::NewRef(ret);
            }

            void force_color() {
                TS_API_AUTO_CHECK(ts_ImageFilter_force_color(m_impl.get()));
            }

            void force_gray() {
                TS_API_AUTO_CHECK(ts_ImageFilter_force_gray(m_impl.get()));
            }

            void force_gray(const std::vector<float> &scale) {
                TS_API_AUTO_CHECK(ts_ImageFilter_force_gray_v2(m_impl.get(), scale.data(), int(scale.size())));
            }

            void force_bgr2gray() {
                force_gray({0.114f, 0.587f, 0.299f});
            }

            void force_rgb2gray() {
                force_gray({0.299f, 0.587f, 0.114f});
            }

            void norm_image(float epsilon) {
                TS_API_AUTO_CHECK(ts_ImageFilter_norm_image(m_impl.get(), epsilon));
            }

            Module module() const {
                auto obj = Module::NewRef(ts_ImageFilter_module(m_impl.get()));
                TS_API_AUTO_CHECK(obj != nullptr);
                return std::move(obj);
            }

        private:
            ImageFilter(raw *ptr) : m_impl(pack(ptr)) {}

            static shared_raw pack(raw *ptr) { return shared_raw(ptr, ts_free_ImageFilter); }

            shared_raw m_impl;
        };
    }
}

#endif //TENNIS_API_CPP_IMAGE_FILTER_H
