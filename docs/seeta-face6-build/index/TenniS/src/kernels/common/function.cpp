//
// Created by yang on 2019/10/22.
//

#include "kernels/common/function.h"
#include "kernels/common/math.h"
#include "kernels/cpu/pad2d_algorithm.h"
#include <algorithm>
#include <array>

namespace ts{

    template <typename T>
    void KernelCommonFunc<T>::in_out_pad_and_fix_size(const Tensor &input,
                                 const Shape &kernel_shape,
                                 const Tensor &out,
                                 int out_h_tile,
                                 int out_w_tile,
                                 const Padding2D &padding,
                                 float padding_value,
                                 const Stride2D &stride,
                                 const KSize2D &ksize,
                                 Tensor &input_padded,
                                 Tensor &out_padded,
                                 bool &out_padded_flag) {

        auto input_shape = input.sizes();
        auto out_shape = out.sizes();

        int num = input_shape[0];
        int input_channel = input_shape[1];
        int input_height = input_shape[2];
        int input_width = input_shape[3];

        int out_channel = out_shape[1];
        int out_height = out_shape[2];
        int out_width = out_shape[3];

        int input_padded_height = input_height + padding.top + padding.bottom;
        int input_padded_width = input_width + padding.left + padding.right;

        int out_padded_height = round_up<int>(out_height, out_h_tile);
        int out_padded_width = round_up<int>(out_width, out_w_tile);

        //input_padded_heigh = std::max(input_padded_heigh, (out_padded_height - 1) * stride.h + (ksize.height - 1) * dilation.h + 1);
        input_padded_height = std::max(input_padded_height, (out_padded_height - 1) * stride.height + ksize.height);
        input_padded_width = std::max(input_padded_width, (out_padded_width - 1) * stride.width + ksize.width);
        //input_padded_height = std::max(input_padded_height, out_padded_height + 2);
        //input_padded_width = std::max(input_padded_width, out_padded_width + 2);

        Padding2D fixed_input_pad;
        //fixed_input_pad.top = (input_padded_height - input_height) >> 1;
        fixed_input_pad.top = padding.top;
        fixed_input_pad.bottom = input_padded_height - input_height - fixed_input_pad.top;
        //fixed_input_pad.left = (input_padded_width - input_width) >> 1;
        fixed_input_pad.left = padding.left;
        fixed_input_pad.right = input_padded_width - input_width - fixed_input_pad.left;

        bool in_need_pad = input_padded_height != input_height || input_padded_width != input_width;
        bool out_need_pad = out_padded_height != out_height || out_padded_width != out_width;
        if (in_need_pad) {
            Tensor padded_input(Tensor::InFlow::HOST, input.dtype(),
                                {num, input_channel, input_padded_height, input_padded_width});
            std::array<int, 2> input_pad_h = {fixed_input_pad.top, fixed_input_pad.bottom};
            std::array<int, 2> input_pad_w = {fixed_input_pad.left, fixed_input_pad.right};
            cpu::PadAlgorithm<T>::pad2d(input, input_pad_h, input_pad_w, (float) padding_value, padded_input);
            input_padded = std::move(padded_input);
        }
        if (out_need_pad) {
            Tensor padded_out(Tensor::InFlow::HOST, out.dtype(),
                              {num, out_channel, out_padded_height, out_padded_width});
            std::array<int, 2> out_pad_h = {0, out_padded_height - out_height};
            std::array<int, 2> out_pad_w = {0, out_padded_width - out_width};
            cpu::PadAlgorithm<T>::pad2d(out, out_pad_h, out_pad_w, (float) padding_value, padded_out);
            out_padded = std::move(padded_out);
        }
        out_padded_flag = out_need_pad;
    }

    template <typename T>
    bool KernelCommonFunc<T>::winograd_check(const Shape &ksize,
                                             const Stride2D &stride,
                                             const Dilation2D &dilation){

        int kernel_height = ksize[2];
        int kernel_width = ksize[3];
        int input_channel = ksize[1];
        int out_channel = ksize[0];

        if(kernel_height == 3 && kernel_width == 3 && stride.width == 1 && stride.height == 1 && dilation.height == 1 && dilation.width == 1){
            if(input_channel >= 32 && out_channel >= 32){
                return true;
            }
        }
        return false;
    }

    template <typename T>
    bool KernelCommonFunc<T>::winograd_mode_select(const Shape &input_shape,
                                                   const int out_channels,
                                                   WinogradConv2DMode& winograd_model){
        int input_channels = input_shape[1];
        int input_height = input_shape[2];
        int input_width = input_shape[3];

        if(input_channels >= 16 && out_channels >= 16){
            if(input_channels * out_channels < 64 * 64){
                if(input_height >= 26 && input_width >= 26){
                    winograd_model = F2X2_3X3;
                    return true;
                }
            }
            else if(input_channels * out_channels < 128 * 128){
                if(input_height < 26 && input_width < 26){
                    return false;
                }
                else if(input_height <= 50 && input_width <= 50){
                    winograd_model = F2X2_3X3;
                    return true;
                }
                else{
                    winograd_model = F6X6_3X3;
                    return true;
                }
            }
            else if(input_channels * out_channels < 256 * 256){
                if(input_height >= 26 && input_width >= 26){
                    if(input_height <= 50 && input_width <= 50){
                        winograd_model = F2X2_3X3;
                        return true;
                    }
                    else{
                        winograd_model = F6X6_3X3;
                        return true;
                    }
                }
            }
            else{
                if(input_height <= 16 && input_width <= 16){
                    winograd_model = F2X2_3X3;
                    return true;
                }
                else{
                    winograd_model = F6X6_3X3;
                    return true;
                }
            }
        }

        return false;
    }

    template <typename T>
    bool KernelCommonFunc<T>::winograd_mode_select_on_arm(const Shape &input_shape,
                                                          const int out_channels,
                                                          WinogradConv2DMode& winograd_model){
        int input_channels = input_shape[1];
        int input_height = input_shape[2];
        int input_width = input_shape[3];

        if(input_channels * out_channels >= 256 * 256){
            if(input_height > 16 && input_width > 16){
                winograd_model = F6X6_3X3;
            }
            else{
                winograd_model = F2X2_3X3;
            }
        }
        else{
            winograd_model = F2X2_3X3;
        }

        return true;

//        if(input_channels >= 16 && out_channels >= 16){
//            if(input_channels * out_channels <= 32 *32){
//                if(input_height >= 18 && input_width >= 18){
//                    winograd_model = F2X2_3X3;
//                    return true;
//                }
//            }
//            else if(input_channels * out_channels >= 256 * 256){
//                if(input_height > 16 && input_width > 16){
//                    winograd_model = F6X6_3X3;
//                    return true;
//                }
//                else{
//                    winograd_model = F2X2_3X3;
//                    return true;
//                }
//            }
//            else{
//                winograd_model = F2X2_3X3;
//                return true;
//            }
//        }
//
//        return false;
    }


} //ts

template class ts::KernelCommonFunc<ts::dtype<ts::FLOAT32>::declare>;
template class ts::KernelCommonFunc<ts::dtype<ts::FLOAT64>::declare>;



