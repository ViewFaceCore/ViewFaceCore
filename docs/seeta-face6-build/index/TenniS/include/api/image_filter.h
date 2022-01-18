//
// Created by keir on 2019/3/16.
//

#ifndef TENNIS_API_IMAGE_FILTER_H
#define TENNIS_API_IMAGE_FILTER_H

#include "common.h"
#include "device.h"
#include "tensor.h"
#include "module.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Filter image before workbench run. Image should be NHWC or HWC format.
 * @see ts_Workbench_bind_filter and ts_Workbench_bind_filter_by_name
 */
struct ts_ImageFilter;
typedef struct ts_ImageFilter ts_ImageFilter;

/**
 * Resize method
 */
enum ts_ResizeMethod {
    TS_RESIZE_BILINEAR = 0,
    TS_RESIZE_BICUBIC = 1,
    TS_RESIZE_NEAREST = 2,
};
typedef enum ts_ResizeMethod ts_ResizeMethod;

/**
 * New ts_new_ImageFilter
 * @param device computing device, NULL means "CPU"
 * @return new ref of ts_ImageFilter
 * @note return NULL if failed
 * @note call @see ts_free_ImageFilter to free ts_ImageFilter
 */
TENNIS_C_API ts_ImageFilter *ts_new_ImageFilter(const ts_Device *device);

/**
 * Free ts_ImageFilter
 * @param filter the ts_ImageFilter ready to be free
 */
TENNIS_C_API void ts_free_ImageFilter(const ts_ImageFilter *filter);

/**
 * Clear set image filter
 * @param filter the return value of ts_new_ImageFilter
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_clear(ts_ImageFilter *filter);

/**
 * Compile set image filters
 * @param filter the return value of ts_new_ImageFilter
 * @return false if failed
 * No need to call explicitly.
 */
TENNIS_C_API ts_bool ts_ImageFilter_compile(ts_ImageFilter *filter);

/**
 * Add filter to stream: to float.
 * @param filter the return value of ts_new_ImageFilter
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_to_float(ts_ImageFilter *filter);

/**
 * Add filter to stream: scale image; multiply f on each pixel.
 * @param filter the return value of ts_new_ImageFilter
 * @param f scale value
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_scale(ts_ImageFilter *filter, float f);

/**
 * Add filter to stream: sub mean value on image; sub mean on each channel.
 * @param filter the return value of ts_new_ImageFilter
 * @param mean mean value
 * @param len length of given mean
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_sub_mean(ts_ImageFilter *filter, const float *mean, int32_t len);

/**
 * Add filter to stream: div std value on image; div std on each channel.
 * @param filter the return value of ts_new_ImageFilter
 * @param std std value
 * @param len length of given std
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_div_std(ts_ImageFilter *filter, const float *std, int32_t len);

/**
 * Add filter to stream: resize image to given [width, height].
 * @param filter the return value of ts_new_ImageFilter
 * @param width new width
 * @param height new height
 * @return false if failed
 * @note using TS_RESIZE_BILINEAR by default
 */
TENNIS_C_API ts_bool ts_ImageFilter_resize(ts_ImageFilter *filter, int32_t width, int32_t height);

/**
 * Add filter to stream: equal scale image short edge to given width.
 * @param filter the return value of ts_new_ImageFilter
 * @param width short edge dest size
 * @return false if failed
 * @note using TS_RESIZE_BILINEAR by default
 */
TENNIS_C_API ts_bool ts_ImageFilter_resize_scalar(ts_ImageFilter *filter, int32_t width);

/**
 * Add filter to stream: center crop image to given [width, height]
 * @param filter the return value of ts_new_ImageFilter
 * @param width wanted width
 * @param height wanted height
 * @return false if failed
 * @note if given image is smaller than [width, height], this filter will zero pad image to [width, height]
 */
TENNIS_C_API ts_bool ts_ImageFilter_center_crop(ts_ImageFilter *filter, int32_t width, int32_t height);

/**
 * Add filter to stream: swap channel
 * @param filter the return value of ts_new_ImageFilter
 * @param shuffle new layout of channel
 * @param len length of given shuffle
 * @return flase if failed
 * @note call ts_ImageFilter_channel_swap(filter, {2, 1, 0}, 3) means do BGR2RGB method
 */
TENNIS_C_API ts_bool ts_ImageFilter_channel_swap(ts_ImageFilter *filter, const int32_t *shuffle, int32_t len);

/**
 * Add filter to stream: final change HWC image to CHW
 * @param filter the return value of ts_new_ImageFilter
 * @return flase if failed
 * @note this filter must be final filter, because that image format changed
 */
TENNIS_C_API ts_bool ts_ImageFilter_to_chw(ts_ImageFilter *filter);

/**
 * Add filter to stream: prewhiten image
 * @param filter the return value of ts_new_ImageFilter
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_prewhiten(ts_ImageFilter *filter);

/**
 * Add filter to stream: equal scale image' long edge to given [width, height], fill outer image area with outer_value
 * @param filter the return value of ts_new_ImageFilter
 * @param width wanted image width
 * @param height wanted image height
 * @return false if failed
 * @note using TS_RESIZE_BILINEAR by default
 */
TENNIS_C_API ts_bool ts_ImageFilter_letterbox(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value);

/**
 * Add filter to stream: adjust image to can be divided by [width, height]
 * @param filter the return value of ts_new_ImageFilter
 * @param width times of width
 * @param height times of height
 * @param padding_value padding value
 * @return false if failed
 * @note Ex: input [800, 600] image, divided [32, 32], will pad image to [800, 608]
 */
TENNIS_C_API ts_bool ts_ImageFilter_divided(ts_ImageFilter *filter, int32_t width, int32_t height, float padding_value);

/**
 * Filter image in given tensor, NHWC format or HWC format
 * @param filter the return value of ts_new_ImageFilter
 * @param tensor ready tensor in NHWC or HWC format
 * @return new reference, filtered image
 */
TENNIS_C_API ts_Tensor *ts_ImageFilter_run(ts_ImageFilter *filter, const ts_Tensor *tensor);

/**
 * Add filter to stream: adjust image to can be divided by [width, height]
 * @param filter the return value of ts_new_ImageFilter
 * @param width times of width
 * @param height times of height
 * @param padding_value padding value
 * @param method @see ts_ResizeMethod
 * @return false if failed
 * @note Ex: input [800, 600] image, divided [32, 32], will pad image to [800, 608]
 */
TENNIS_C_API ts_bool ts_ImageFilter_letterbox_v2(ts_ImageFilter *filter, int32_t width, int32_t height, float outer_value,
                                                 ts_ResizeMethod method);
/**
 * Add filter to stream: resize image to given [width, height].
 * @param filter the return value of ts_new_ImageFilter
 * @param width new width
 * @param height new height
 * @param method @see ts_ResizeMethod
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_resize_v2(ts_ImageFilter *filter, int32_t width, int32_t height,
                                              ts_ResizeMethod method);

/**
 * Add filter to stream: equal scale image short edge to given width.
 * @param filter the return value of ts_new_ImageFilter
 * @param width short edge dest size
 * @param method @see ts_ResizeMethod
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_resize_scalar_v2(ts_ImageFilter *filter, int32_t width,
                                                     ts_ResizeMethod method);

/**
 * Add filter to stream: force image to color mode, may copy each gary channel to output channels
 * @param filter the return value of ts_new_ImageFilter
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_force_color(ts_ImageFilter *filter);

/**
 * Add filter to stream: force image to gray mode, use [0.114, 0.587, 0.299] converting BGR mode.
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_force_gray(ts_ImageFilter *filter);

/**
 * Add filter to stream: force image to gray mode
 * @param filter the return value of ts_new_ImageFilter
 * @param scale scale value
 * @param len length of given scale
 * @return false if failed
 * @note if scale is NULL or len is 0, this function same as ts_ImageFilter_force_gray
 * @note use [0.114, 0.587, 0.299] converting BGR mode.
 * @note use [0.299, 0.587, 0.114] converting RGB mode.
 */
TENNIS_C_API ts_bool ts_ImageFilter_force_gray_v2(ts_ImageFilter *filter, const float *scale, int32_t len);

/**
 * Add filter to stream: norm image; x = (x - mean) / (std_dev + epsilon)
 * @param filter the return value of ts_new_ImageFilter
 * @param epsilon epsilon value
 * @return false if failed
 */
TENNIS_C_API ts_bool ts_ImageFilter_norm_image(ts_ImageFilter *filter, float epsilon);

/**
 * Get module
 * @param filter the return value of ts_new_ImageFilter
 * @return new reference of ts_Module
 */
TENNIS_C_API ts_Module *ts_ImageFilter_module(const ts_ImageFilter *filter);


#ifdef __cplusplus
}
#endif

#endif //TENNIS_API_IMAGE_FILTER_H
