#include <kernels/cpu/transpose_conv2d_core.h>
#include <core/tensor_builder.h>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <core/device.h>

#include "backend/common_structure.h"

#include <kernels/cpu/math_cpu.h>
#include <kernels/cpu/im2col.h>

#include <fstream>
#include <frontend/intime.h>
//#include "kernels/common/simd.h"
#ifdef TS_USE_CBLAS
#include <kernels/cblas/math_cblas.h>
#endif

/////////////////////////////////////////////////
namespace ts {
    namespace cpu {
        template<typename T>
        static void
        cpu_transpose_conv2d_nchw_compute_run(const Tensor &x, const Padding2D &padding, float padding_value,
                                              const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                              Tensor &out, Stack &stack, bool kernel_packed) {
            if (kernel_packed) {
                TS_LOG_ERROR << "No transpose_conv2d packed method defined." << eject;
            }

            auto weight_shape = w.sizes();
            auto output_shape = out.sizes();
            auto x_shape = x.sizes();
            int kernel_dims = weight_shape[1] * weight_shape[2] * weight_shape[3];
            int conv_out_spatial_dim = x_shape[2] * x_shape[3];
            int output_number_offset = output_shape[1] * output_shape[2] * output_shape[3];
            int input_number_offset = x_shape[1] * conv_out_spatial_dim;
            int col_buffer_size = weight_shape[1] * weight_shape[2] * weight_shape[3] * x_shape[2] * x_shape[3];

            auto number = x_shape[0];
            auto input_channels = weight_shape[1];
            Size2D ksize(weight_shape[2], weight_shape[3]);
            Size2D input(output_shape[2], output_shape[3]);


#ifdef TS_USE_CBLAS
            const T *pinput = x.data<T>();
            const T *pweight = w.data<T>();
            T *poutput = out.data<T>();
#else
            const T *pinput = x.data<T>();
            T *poutput = out.data<T>();

            auto transposed_weight = intime::transpose(w.reshape({w.size(0), -1}), {1, 0});
            auto transposed_pweight = transposed_weight.data<T>();
#endif

            Tensor col_tensor;
            T *col_buffer = nullptr;

            bool is_1x1_conv = stride.height == 1 && stride.width == 1 &&
                               ksize.height == 1 && ksize.width == 1 &&
                               padding.top == 0 && padding.bottom == 0 &&
                               padding.left == 0 && padding.right == 0;

            // 1x1 conv do not need im2col
            Shape col_shape;
            col_shape.resize(1);
            col_shape[0] = col_buffer_size;
            col_tensor = stack.make(out.dtype(), col_shape, MemoryDevice(CPU));
            col_buffer = col_tensor.data<T>();

            for (int i = 0; i < number; i++) {
#ifdef TS_USE_CBLAS
                cblas::math<T>::gemm(ts::blas::Trans, ts::blas::NoTrans, kernel_dims, conv_out_spatial_dim,
                                     weight_shape[0], 1.0, pweight, pinput, 0, col_buffer);
#else
                cpu::math<T, T>::gemm(kernel_dims, conv_out_spatial_dim,
                                      weight_shape[0], 1.0, transposed_pweight, pinput, 0, col_buffer, true, true);
#endif


                if (is_1x1_conv) {
                    std::memcpy(poutput, col_buffer, sizeof(T) * col_buffer_size);
                } else {
                    col2im_cpu((const T *) col_buffer, input_channels, input.height, input.width,
                               ksize.height, ksize.width,
                               padding.top, padding.bottom,
                               padding.left, padding.right,
                               stride.height, stride.width,
                               dilation.height, dilation.width,
                               poutput);

                }

                pinput += input_number_offset;
                poutput += output_number_offset;

            }
        }

        void Conv2DTransposeCore::conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                                   const Tensor &w,
                                                   const Stride2D &stride, const Dilation2D &dilation,
                                                   Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) {
            if (format != FORMAT_NCHW) {
                TS_LOG_ERROR << "Conv2DTransposeCore only support NCHW" << eject;
            }
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_transpose_conv2d_nchw_compute_run<TYPE>(x, padding, padding_value, w, stride, dilation, out, stack, kernel_packed); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Conv2DTransposeCore not support this data type: " << "(" << dtype << ")"
                                 << type_str(dtype) << eject;
                    break;
                }
            }
            return;
        }

    }

}

/////////////////////////////////////////////////

//using namespace ts;
//using namespace cpu;
//TS_REGISTER_OPERATOR(Transpose_Conv2D, CPU, name::layer::add_bias())

//TS_REGISTER_OPERATOR(Transpose_Conv2D, CPU, std::string("transpose_conv2d"))

