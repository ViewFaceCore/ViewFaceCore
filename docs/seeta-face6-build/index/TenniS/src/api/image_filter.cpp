//
// Created by kier on 2019/3/16.
//

#include <api/image_filter.h>

#include "declare_image_filter.h"
#include "declare_tensor.h"
#include "declare_module.h"

using namespace ts;

ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device) {
    TRY_HEAD
    //if (!device) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_ImageFilter> image_filter(
            device
            ? new ts_ImageFilter(ComputingDevice(device->type, device->id))
            : new ts_ImageFilter()
    );
    RETURN_OR_CATCH(image_filter.release(), nullptr)
}

void ts_free_ImageFilter(const ts_ImageFilter *filter) {
    TRY_HEAD
    delete filter;
    TRY_TAIL
}

ts_bool ts_ImageFilter_clear(ts_ImageFilter *filter) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->clear();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_compile(ts_ImageFilter *filter) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->compile();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_to_float(ts_ImageFilter *filter) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->to_float();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_scale(ts_ImageFilter *filter, float f) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->scale(f);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_sub_mean(ts_ImageFilter *filter, const float *mean, int32_t len) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    if (!mean) throw Exception("NullPointerException: @param: 2");
    (*filter)->sub_mean(std::vector<float>(mean, mean + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_div_std(ts_ImageFilter *filter, const float *std, int32_t len) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    if (!std) throw Exception("NullPointerException: @param: 2");
    (*filter)->div_std(std::vector<float>(std, std + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int32_t width, int32_t height) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->resize(width, height, ImageFilter::ResizeMethod::BILINEAR);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_resize_scalar(ts_ImageFilter *filter, int32_t width) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->resize(width, ImageFilter::ResizeMethod::BILINEAR);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int32_t width, int32_t height) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->center_crop(width, height);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int32_t *shuffle, int32_t len) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    if (!shuffle) throw Exception("NullPointerException: @param: 2");
    (*filter)->channel_swap(std::vector<int32_t>(shuffle, shuffle + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter) {
    TRY_HEAD
    if (!filter) throw Exception("NullPointerException: @param: 1");
    (*filter)->to_chw();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_prewhiten(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->prewhiten();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_Tensor *ts_ImageFilter_run(ts_ImageFilter *filter, const ts_Tensor *tensor) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        if (!tensor) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                (*filter)->run(**tensor)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_ImageFilter_letterbox(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->letterbox(width, height, outer_value, ImageFilter::ResizeMethod::BILINEAR);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_divided(ts_ImageFilter *filter, int32_t width, int32_t height, float padding_value) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->divided(width, height, padding_value);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_resize_v2(ts_ImageFilter *filter, int32_t width, int32_t height,
        ts_ResizeMethod method) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->resize(width, height, ImageFilter::ResizeMethod(int32_t(method)));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_resize_scalar_v2(ts_ImageFilter *filter, int32_t width,
        ts_ResizeMethod method) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->resize(width, ImageFilter::ResizeMethod(int32_t(method)));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_letterbox_v2(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value,
        ts_ResizeMethod method) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->letterbox(width, height, outer_value, ImageFilter::ResizeMethod(int32_t(method)));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_force_color(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->force_color();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_force_gray(ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->force_gray();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_force_gray_v2(ts_ImageFilter *filter, const float *scale, int32_t len) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        if (!scale) throw Exception("NullPointerException: @param: 2");
        (*filter)->force_gray(std::vector<float>(scale, scale + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_ImageFilter_norm_image(ts_ImageFilter *filter, float epsilon) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        (*filter)->norm_image(epsilon);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_Module *ts_ImageFilter_module(const ts_ImageFilter *filter) {
    TRY_HEAD
        if (!filter) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Module> module(new ts_Module(
                (*filter)->module()
                ));
    RETURN_OR_CATCH(module.release(), nullptr)
}
