//
// Created by kier on 2019/1/12.
//

#include <backend/name.h>

#include "backend/name.h"

namespace ts {
    namespace name {
        namespace layer {
            const string &field() TS_NOEXCEPT { static string str = "_field"; return str; }
            const string &pack() TS_NOEXCEPT { static string str = "_pack"; return str; }
            const string &dimshuffle() TS_NOEXCEPT { static string str = "_dimshuffle"; return str; }
            const string &transpose() TS_NOEXCEPT { static string str = "_transpose"; return str; }
            const string &reshape() TS_NOEXCEPT { static string str = "_reshape"; return str; }
            const string &conv2d() TS_NOEXCEPT { static string str = "conv2d"; return str; }
            const string &conv2d_v2() TS_NOEXCEPT { static string str = "conv2d_v2"; return str; }
            const string &shape() TS_NOEXCEPT { static string str = "_shape"; return str; }
            const string &pad() TS_NOEXCEPT { static string str = "pad"; return str; }
            const string &depthwise_conv2d() TS_NOEXCEPT { static string str = "depthwise_conv2d"; return str; }
            const string &depthwise_conv2d_v2() TS_NOEXCEPT { static string str = "depthwise_conv2d_v2"; return str; }
            const string &add_bias() TS_NOEXCEPT { static string str = "add_bias"; return str; }
            const string &batch_norm() TS_NOEXCEPT { static string str = "batch_norm"; return str; }
            const string &batch_scale() TS_NOEXCEPT { static string str = "batch_scale"; return str; }
            const string &fused_batch_norm() TS_NOEXCEPT { static string str = "fused_batch_norm"; return str; }
            const string &add() TS_NOEXCEPT { static string str = "add"; return str; }
            const string &sub() TS_NOEXCEPT { static string str = "sub"; return str; }
            const string &mul() TS_NOEXCEPT { static string str = "mul"; return str; }
            const string &div() TS_NOEXCEPT { static string str = "div"; return str; }
            const string &inner_prod() TS_NOEXCEPT { static string str = "inner_prod"; return str; }
            const string &relu() TS_NOEXCEPT { static string str = "relu"; return str; }
            const string &prelu() TS_NOEXCEPT { static string str = "prelu"; return str; }
            const string &relu_max() TS_NOEXCEPT { static string str = "relu_max"; return str; }
            const string &sigmoid() TS_NOEXCEPT { static string str = "sigmoid"; return str; }
            const string &softmax() TS_NOEXCEPT { static string str = "softmax"; return str; }
            const string &concat() TS_NOEXCEPT { static string str = "concat"; return str; }
            const string &flatten() TS_NOEXCEPT { static string str = "flatten"; return str; }
            const string &to_float() TS_NOEXCEPT { static string str = "to_float"; return str; }
            const string &pooling2d() TS_NOEXCEPT { static string str = "pooling2d"; return str; }
            const string &pooling2d_v2() TS_NOEXCEPT { static string str = "pooling2d_v2"; return str; }
            const string &resize2d() TS_NOEXCEPT { static string str = "_resize2d"; return str; }
            const string &mx_pooling2d_padding() TS_NOEXCEPT { static string str = "_mx_pooling2d_padding"; return str; }
            const string &nhwc_center_crop2d() TS_NOEXCEPT { static string str = "_nhwc_center_crop2d"; return str; }
            const string &cast() TS_NOEXCEPT { static string str = "_cast"; return str; }
            const string &onnx_pooling2d_padding() TS_NOEXCEPT { static string str = "_onnx_pooling2d_padding"; return str; }
            const string &gather() TS_NOEXCEPT { static string str = "gather"; return str; }
            const string &unsqueeze() TS_NOEXCEPT { static string str = "unsqueeze"; return str; }
            const string &gemm() TS_NOEXCEPT { static string str = "gemm"; return str; }
            const string &reshape_v2() TS_NOEXCEPT { static string str = "_reshape_v2"; return str; }
            const string &global_pooling2d() TS_NOEXCEPT { static string str = "global_pooling2d"; return str; }
            const string &limit() TS_NOEXCEPT { static string str = "_limit"; return str; }

            const string &shape_index_patch() TS_NOEXCEPT { static string str = "shape_index_patch"; return str; }

            const string &tf_pooling2d_padding() TS_NOEXCEPT { static string str = "_tf_pooling2d_padding"; return str; }

            const string &tf_conv2d_padding() TS_NOEXCEPT { static string str = "_tf_conv2d_padding"; return str; }
            const string &nhwc_scale_resize2d() TS_NOEXCEPT { static string str = "_nhwc_scale_resize2d"; return str; }

            const string &strided_slice() TS_NOEXCEPT { static string str = "strided_slice"; return str; }
            const string &stack() TS_NOEXCEPT { static string str = "stack"; return str; }

            const string &crop_nd() TS_NOEXCEPT { static string str = "crop_nd"; return str; }
            const string &dcn_v2_forward() TS_NOEXCEPT { static string str = "dcn_v2_forward"; return str; }

            const string &chunk() TS_NOEXCEPT { static string str = "chunk"; return str; }

            const string &transpose_conv2d() TS_NOEXCEPT { static string str = "transpose_conv2d"; return str; }
            const string &batchtospace4d() TS_NOEXCEPT { static string str = "batch_to_space4d"; return str; }
            const string &spacetobatch4d() TS_NOEXCEPT { static string str = "space_to_batch4d"; return str; }

            const string &affine_sample2d() TS_NOEXCEPT { static string str = "affine_sample2d"; return str; }

            const string &squeeze() TS_NOEXCEPT { static string str = "squeeze"; return str; }

            const string &gatherv2() TS_NOEXCEPT { static string str = "gatherv2"; return str; }
            const string &resize_nearest_neighbor() TS_NOEXCEPT { static string str = "resize_nearest_neighbor"; return str; }
            const string &rsqrt() TS_NOEXCEPT { static string str = "rsqrt"; return str; }
            const string &maximum() TS_NOEXCEPT { static string str = "maximum"; return str; }
            const string &max() TS_NOEXCEPT { static string str = "max"; return str; }
            const string &square() TS_NOEXCEPT { static string str = "square"; return str; }

            const string &range() TS_NOEXCEPT { static string str = "range"; return str; }
            const string &exp() TS_NOEXCEPT { static string str = "exp"; return str; }
            const string &slice() TS_NOEXCEPT { static string str = "slice"; return str; }
            const string &argmax() TS_NOEXCEPT { static string str = "argmax"; return str; }
            const string &non_max_suppression_v3() TS_NOEXCEPT { static string str = "non_max_suppression_v3"; return str; }
            const string &topkv2() TS_NOEXCEPT { static string str = "topkv2"; return str; }

            const string &prewhiten() TS_NOEXCEPT { static string str = "prewhiten"; return str; }

            const string &copy() TS_NOEXCEPT {
                static string str = "_copy";
                return str;
            }

            const string &winograd_transform_kernel() TS_NOEXCEPT { static string str = "winograd_transform_kernel"; return str; }
            const string &conv2d_winograd() TS_NOEXCEPT { static string str = "conv2d_winograd"; return str; }

            const string &nhwc_letterbox() TS_NOEXCEPT { static string str = "_nhwc_letterbox"; return str; }
            const string &sample2d() TS_NOEXCEPT { static string str = "sample2d"; return str; }
            const string &divided() TS_NOEXCEPT { static string str = "divided"; return str; }

            const string &yolo() TS_NOEXCEPT { static string str = "yolo"; return str; }
            const string &yolo_poster() TS_NOEXCEPT { static string str = "yolo_poster"; return str; }

            const string &l2_norm() TS_NOEXCEPT { static string str = "l2_norm"; return str; }

            // 2019-06-27
            const string &force_color() TS_NOEXCEPT { static string str = "force_color"; return str; }
            const string &force_gray() TS_NOEXCEPT { static string str = "force_gray"; return str; }

            const string &norm_image() TS_NOEXCEPT { static string str = "norm_image"; return str; }

            const string &quantize() TS_NOEXCEPT { static string str = "quantize"; return str; }
            const string &conv2d_quantized() TS_NOEXCEPT { static string str = "conv2d_quantized"; return str; }

            const string &reduce_sum() TS_NOEXCEPT { static string str = "reduce_sum"; return str; }
            const string &reduce_mean() TS_NOEXCEPT { static string str = "reduce_mean"; return str; }
            const string &sqrt() TS_NOEXCEPT { static string str = "sqrt"; return str; }
            const string &tile() TS_NOEXCEPT { static string str = "tile"; return str; }

            const string &roi_align() TS_NOEXCEPT { static string str = "roi_align"; return str; }
            const string &proposal() TS_NOEXCEPT { static string str = "proposal"; return str; }

            const string &conv2d_winograd_v2() TS_NOEXCEPT { static string str = "conv2d_winograd_v2"; return str; }
        }

        namespace typo {
            string dialations = "dialations";
        }

        string NCHW = "NCHW";
        string NHWC = "NHWC";
        string dim = "dim";
        string shuffle = "shuffle";
        string value = "value";
        string permute = "permute";
        string shape = "shape";
        string format = "format";
        string padding = "padding";
        string padding_value = "padding_value";
        string stride = "stride";
        string dilation = "dilation";
        string kernel_packed = "kernel_packed";
        string epsilon = "epsilon";
        string max = "max";
        string slope = "slope";
        string type = "type";
        string padding_type = "padding_type";
        string ksize = "ksize";
        string valid = "valid";
        string device = "device";
        string offset = "offset";
		string smooth = "smooth";
        string size = "size";
        string prewhiten = "prewhiten";
        string dtype = "dtype";
        string output_shape = "output_shape";

        string auto_pad = "auto_pad";
        string axis = "axis";
        string axes = "axes";
        string NOTSET = "NOTSET";
        string SAME_UPPER = "SAME_UPPER";
        string SAME_LOWER = "SAME_LOWER";
        string VALID = "VALID";
        string alpha = "alpha";
        string beta = "beta";
        string transA = "transA";
        string transB = "transB";

        string padding_method = "padding_method";
        string SAME = "SAME";

        string begin = "begin";
        string end = "end";

        string shift = "shift";
        
        string chunks = "chunks";
        string deformable_groups = "deformable_groups";
        
        string crop = "crop";
        string block_shape = "block_shape";

        string sorted = "sorted";
        string number = "number";
        string mode = "mode";
        string iou_threshold = "iou_threshold";
        string score_threshold = "score_threshold";

        string max_output_size = "max_output_size";
        string align_corners = "align_corners";

        string keep_dims = "keep_dims";

        string winograd_mode = "winograd_mode";
        string winograd_f23 = "winograd_f23";
        string winograd_f63 = "winograd_f63";

        string outer_value = "outer_value";
        string scale = "scale";

        string quantize_scale = "quantize_scale";
        string dequantize_scales = "dequantize_scales";

        string dims = "dims";
        string repeats = "repeats";

        string transpose = "transpose";

        string kernel_winograd_transformed = "kernel_winograd_transformed";
    }
}
