#include <kernels/cpu/depthwise_conv2d_core.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <kernels/cpu/depthwise_conv2d_algorithm.h>


namespace ts {
    namespace cpu {
        template<typename T>
        static void
        cpu_depthwise_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                              const Tensor &weight, const Stride2D &stride, const Dilation2D &dilation,
                                              Tensor &out, Stack &stack, bool kernel_packed) {
            if (kernel_packed) {
                TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
            }

            auto weight_shape = weight.sizes();
            auto dtype = x.dtype();
            if (dtype != FLOAT32) {
                DepthwiseConv2dAlgorithm<T>::depthwise_general(x, padding, padding_value, weight, stride, dilation, out);
                return;
            }

            if (weight_shape[2] == 3 && weight_shape[3] == 3 && stride.height == 1 && stride.width == 1 && dilation.height == 1 && dilation.width == 1) {
                DepthwiseConv2dAlgorithm<T>::depthwise_3x3_s1(x, padding, padding_value, weight, stride, dilation, out);
            }

            else if (weight_shape[2] == 3 && weight_shape[3] == 3 && stride.height == 2 && stride.width == 2 && dilation.height == 1 && dilation.width == 1) {
                DepthwiseConv2dAlgorithm<T>::depthwise_3x3_s2(x, padding, padding_value, weight, stride, dilation, out);
            }

            else {
                DepthwiseConv2dAlgorithm<T>::depthwise_general(x, padding, padding_value, weight, stride, dilation, out);
            }
        }

        void
        DepthwiseConv2DCore::conv2d(const Tensor &x, const Padding2D &padding, float padding_value, const Tensor &w,
                                    const Stride2D &stride, const Dilation2D &dilation, Conv2DFormat format,
                                    Tensor &out, Stack &stack, bool kernel_packed) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "DepthwiseConv2D only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_depthwise_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack, kernel_packed); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "DepthwiseConv2D not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}
