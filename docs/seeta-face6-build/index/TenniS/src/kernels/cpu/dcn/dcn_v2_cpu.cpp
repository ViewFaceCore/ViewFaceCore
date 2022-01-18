
#include "dcn_v2.h"
#include "dcn_v2_im2col_cpu.h"

#include "utils.h"
#include "core/device_context.h"

#include "utils/ctxmgr_lite.h"

using scalar_t = float;

namespace ts {
    static void createBatchGemmBuffer(const float **input_b, float **output_b,
                               const float **columns_b, const float **ones_b,
                               const float **weight_b, const float **bias_b,
                               const float *input, float *output,
                               const float *columns, const float *ones,
                               const float *weight, const float *bias,
                               const int input_stride, const int output_stride,
                               const int columns_stride, const int ones_stride,
                               const int num_batches) {
        // const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = 0;
        while (idx < num_batches) {
            input_b[idx] = input + idx * input_stride;
            output_b[idx] = output + idx * output_stride;
            columns_b[idx] = columns + idx * columns_stride;
            ones_b[idx] = ones + idx * ones_stride;
            // share weights and bias within a Mini-Batch
            weight_b[idx] = weight;
            bias_b[idx] = bias;
            ++idx;
        }
    }

    Tensor
    dcn_v2_cpu_forward(const Tensor &input,
                       const Tensor &weight,
                       const Tensor &bias,
                       const Tensor &offset,
                       const Tensor &mask,
                       const int kernel_h,
                       const int kernel_w,
                       const int stride_h,
                       const int stride_w,
                       const int pad_h,
                       const int pad_w,
                       const int dilation_h,
                       const int dilation_w,
                       const int deformable_group,
                       Tensor *buffer_output) {
        using scalar_t = float;
        TS_AUTO_CHECK(input.dtype() == FLOAT32);

        const int batch = input.size(0);
        const int channels = input.size(1);
        const int height = input.size(2);
        const int width = input.size(3);

        const int channels_out = weight.size(0);
        const int channels_kernel = weight.size(1);
        const int kernel_h_ = weight.size(2);
        const int kernel_w_ = weight.size(3);

        // printf("Kernels: %d %d %d %d\n", kernel_h_, kernel_w_, kernel_w, kernel_h);
        // printf("Channels: %d %d\n", channels, channels_kernel);
        // printf("Channels: %d %d\n", channels_out, channels_kernel);

        TS_AUTO_CHECK(kernel_h_ == kernel_h && kernel_w_ == kernel_w)
                ("Input shape and kernel shape wont match: (")
                (kernel_h_)(" x ")(kernel_w)(" vs. ")(kernel_h_)(" x ")(kernel_w_)(").");

        TS_AUTO_CHECK(channels == channels_kernel)
                ("Input shape and kernel channels wont match: (")
                (channels)(" vs. ")(channels_kernel)(").");

        const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        auto ones = dcn::cpu::ones(input.dtype(), {batch, height_out, width_out});
        auto columns = dcn::cpu::empty(input.dtype(),
                                       {batch, channels * kernel_h * kernel_w, 1 * height_out * width_out});
        Tensor output;
        if (buffer_output != nullptr) {
            output = *buffer_output;
            TS_AUTO_CHECK(output.device() == input.device() && output.proto() == Tensor::Prototype(input.dtype(),
                                                                                                   {batch, channels_out,
                                                                                                    height_out,
                                                                                                    width_out}));
        } else {
            output = dcn::cpu::empty(input.dtype(), {batch, channels_out, height_out, width_out});
        }

        // prepare for batch-wise computing, which is significantly faster than instance-wise computing
        // when batch size is large.
        // launch batch threads
        int matrices_size = batch * sizeof(float *);
        std::vector<SyncMemory> bufer_b(6);
        auto input_b = static_cast<const float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[0]));
        auto output_b = static_cast<float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[1]));
        auto columns_b = static_cast<const float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[2]));
        auto ones_b = static_cast<const float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[3]));
        auto weight_b = static_cast<const float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[4]));
        auto bias_b = static_cast<const float **>(dcn::cpu::CPUMalloc(input.device(), matrices_size, bufer_b[5]));

        createBatchGemmBuffer(
                input_b, output_b,
                columns_b, ones_b,
                weight_b, bias_b,
                input.data<scalar_t>(),
                output.data<scalar_t>(),
                columns.data<scalar_t>(),
                ones.data<scalar_t>(),
                weight.data<scalar_t>(),
                bias.data<scalar_t>(),
                channels * width * height,
                channels_out * width_out * height_out,
                channels * kernel_h * kernel_w * height_out * width_out,
                height_out * width_out,
                batch);

        long m_ = channels_out;
        long n_ = height_out * width_out;
        long k_ = 1;
        dcn::cpu::CBlas_SgemmBatched('t',
                                     'n',
                                     n_,
                                     m_,
                                     k_,
                                     1.0f,
                                     ones_b, k_,
                                     bias_b, k_,
                                     0.0f,
                                     output_b, n_,
                                     batch);

        modulated_deformable_im2col_cpu(input.data<scalar_t>(),
                                        offset.data<scalar_t>(),
                                        mask.data<scalar_t>(),
                                        batch, channels, height, width,
                                        height_out, width_out, kernel_h, kernel_w,
                                        pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                                        deformable_group,
                                        columns.data<scalar_t>());

        long m = channels_out;
        long n = height_out * width_out;
        long k = channels * kernel_h * kernel_w;
        dcn::cpu::CBlas_SgemmBatched('n',
                                     'n',
                                     n,
                                     m,
                                     k,
                                     1.0f,
                                     (const float **) columns_b, n,
                                     weight_b, k,
                                     1.0f,
                                     output_b, n,
                                     batch);
        return output;
    }
}