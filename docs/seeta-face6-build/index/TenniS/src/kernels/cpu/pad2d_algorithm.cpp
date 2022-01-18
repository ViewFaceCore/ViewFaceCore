#include <kernels/cpu/pad2d_algorithm.h>
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif
#include <array>

namespace ts{
    namespace cpu{

        //NOTE:Only supports pad or cut on NCHW or NHWC formats.
        template<typename T>
        void PadAlgorithm<T>::pad_nchw_nhwc(const Tensor &x,
                             const std::vector<std::array<int, 2>> &padding,
                             float padding_value,
                             Tensor &out){
            const T* input_ptr = x.data<T>();
            T* out_ptr = out.data<T>();

            auto input_shape = x.sizes();
            int num = input_shape[0];
            int input_dim1 = input_shape[1];
            int input_dim2 = input_shape[2];
            int input_tile = input_shape[3];

            auto output_shape = out.sizes();
            int out_dim1 = output_shape[1];
            int out_dim2 = output_shape[2];
            int out_tile = output_shape[3];

            int input_dim1_offset = input_dim2 * input_tile;
            int in_num_offset = input_dim1 * input_dim1_offset;
            int out_dim1_offset = out_dim2 * out_tile;
            int out_num_offset = out_dim1 * out_dim1_offset;

            int dim1 = padding[1][0] < 0 ? input_dim1 + padding[1][0] : input_dim1;
            dim1 = padding[1][1] < 0 ? dim1 + padding[1][1] : dim1;
            int dim2 = padding[2][0] < 0 ? input_dim2 + padding[2][0] : input_dim2;
            dim2 = padding[2][1] < 0 ? dim2 + padding[2][1] : dim2;
            int tile = padding[3][0] < 0 ? input_tile + padding[3][0] : input_tile;
            tile = padding[3][1] < 0 ? tile + padding[3][1] : tile;

            std::array<int,4> input_padding;
            std::array<int,4> output_padding;

            input_padding[0] = padding[0][0] >= 0 ? 0 : -padding[0][0];
            input_padding[1] = padding[1][0] >= 0 ? 0 : -padding[1][0];
            input_padding[2] = padding[2][0] >= 0 ? 0 : -padding[2][0];
            input_padding[3] = padding[3][0] >= 0 ? 0 : -padding[3][0];
            output_padding[0] = padding[0][0] < 0 ? 0 : padding[0][0];
            output_padding[1] = padding[1][0] < 0 ? 0 : padding[1][0];
            output_padding[2] = padding[2][0] < 0 ? 0 : padding[2][0];
            output_padding[3] = padding[3][0] < 0 ? 0 : padding[3][0];

            T value = T(padding_value);
            std::fill(out_ptr, out_ptr + out.count(), value);

            for (int n = 0; n < num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int i = 0; i < dim1; ++i) {
                    for (int j = 0; j < dim2; ++j) {
                        const int input_offset = (n + input_padding[0]) * in_num_offset
                                                 + (i + input_padding[1]) * input_dim1_offset
                                                 + (j + input_padding[2]) * input_tile
                                                 + input_padding[3];
                        const int output_offset = (n + output_padding[0]) * out_num_offset
                                                  + (i + output_padding[1]) * out_dim1_offset
                                                  + (j + output_padding[2]) * out_tile
                                                  + output_padding[3];
                        if(tile < 12){
                            T* out_at = out_ptr + output_offset;
                            const T* input_at = input_ptr + input_offset;
                            for (int l = 0; l < tile; ++l) {
                                out_at[l] = input_at[l];
                            }
                        }
                        else{
                            std::memcpy(out_ptr + output_offset,
                                        input_ptr + input_offset,
                                        tile * sizeof(T));
                        }
                    }
                }
            }
        };

        template<typename T>
        void PadAlgorithm<T>::pad2d(const Tensor &x,
                                     const std::array<int, 2> &padding_h,
                                     const std::array<int, 2> &padding_w,
                                     float padding_value,
                                     Tensor &out){
            int pad_top = padding_h[0];
            int pad_bottom = padding_h[1];
            int pad_left = padding_w[0];
            int pad_right = padding_w[1];

            auto src_shape = x.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = out.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + pad_top + pad_bottom;
            int out_w = src_w + pad_left + pad_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            out.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = x.data<T>();
            T* dst_data = out.data<T>();
            T value = T(padding_value);
            // auto out_device = out.device();
            // auto x_device = x.device();

            for (int n = 0; n < num; n++){
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++){
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    //fill top
                    int h = 0;
                    for (; h < pad_top; h++){
                        int w = 0;
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                    }

                    //fill center
                    for (; h < pad_top + src_shape[2]; h++){
                        int w = 0;
                        for (; w < pad_left; w++){
                            dst_at[w] = value;
                        }
                        //for (; w < (pad_left + src_w); w++) {
                        //    dst_at[w] = src_at[w - pad_left];
                        //}
                        //NOTE:12 is an empirical data,but ts::memcpy is very slow,sad
                        if (src_w < 12){
                            for (; w < (pad_left + src_w); w++){
                                dst_at[w] = src_at[w - pad_left];
                            }
                        }
                        else{
                            //memcpy(dst_at + pad_left, out_device, size_t(src_w * out.proto().type_bytes()),
                            //    src_at, x_device, size_t(src_w * x.proto().type_bytes()));
                            std::memcpy(dst_at + pad_left, src_at, src_w * sizeof(T));
                            w += src_w;
                        }
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                        src_at += src_w;
                    }
                    // fill bottom
                    for (; h < out_h; h++){
                        int w = 0;
                        for (; w < out_w; w++){
                            dst_at[w] = value;
                        }
                        dst_at += out_w;
                    }
                }
            }
        }

        template<typename T>
        void PadAlgorithm<T>::cut2d(const Tensor &x,
                                     const std::array<int, 2> &padding_h,
                                     const std::array<int, 2> &padding_w,
                                     float padding_value,
                                     Tensor &out){
            int cut_top = padding_h[0];
            int cut_bottom = padding_h[1];
            int cut_left = padding_w[0];
            int cut_right = padding_w[1];

            auto src_shape = x.sizes();
            int num = src_shape[0];
            int channel = src_shape[1];
            int src_h = src_shape[2];
            int src_w = src_shape[3];
            int src_channel_offset = src_h * src_w;
            int src_num_offset = src_shape[1] * src_channel_offset;

            auto dst_shape = out.sizes();
            dst_shape[0] = num;
            dst_shape[1] = channel;
            int out_h = src_h + cut_top + cut_bottom;
            int out_w = src_w + cut_left + cut_right;
            dst_shape[2] = out_h;
            dst_shape[3] = out_w;
            out.reshape(dst_shape);
            int dst_channel_offset = out_h * out_w;
            int dst_num_offset = channel * dst_channel_offset;

            const T* src_data = x.data<T>();
            T* dst_data = out.data<T>();

            for (int n = 0; n < num; n++) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int c = 0; c < channel; c++) {
                    const T* src_at = src_data + n * src_num_offset + c * src_channel_offset;
                    T* dst_at = dst_data + n * dst_num_offset + c * dst_channel_offset;

                    const T* cut_at = src_at - cut_top * src_w - cut_left;

                    for (int h = 0; h < out_h; h++) {
                        //for (int w = 0; w < out_w; w++)
                        //{
                        //    dst_at[w] = cut_at[w];
                        //}
                        //NOTE:ts::memcpy is very slow
                        if (out_w <12) {
                            for (int w = 0; w < out_w; w++)
                            {
                                dst_at[w] = cut_at[w];
                            }
                        }
                        else {
                            //memcpy(dst_at, out.device(), size_t(out_w * out.proto().type_bytes()),
                            //cut_at, x.device(), size_t(out_w * x.proto().type_bytes()));
                            std::memcpy(dst_at, cut_at, out_w * sizeof(T));
                        }

                        dst_at += out_w;
                        cut_at += src_w;
                    }
                }
            }
        }

    }//cpu
}//ts

template class ts::cpu::PadAlgorithm<ts::dtype<ts::INT8>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::UINT8>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::INT16>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::UINT16>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::INT32>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::UINT32>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::INT64>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::UINT64>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
template class ts::cpu::PadAlgorithm<ts::dtype<ts::FLOAT64>::declare>;