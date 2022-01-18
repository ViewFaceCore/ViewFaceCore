#include <core/tensor_builder.h>
#include <kernels/cpu/math_cpu.h>
#include <kernels/cblas/math_cblas.h>
#include "runtime/inside/thread_pool.h"
#include "utils/ctxmgr_lite.h"
#include "utils/box.h"

#include "core/tensor.h"

namespace ts {
    namespace cpu {
        template<typename T>
        class TS_DEBUG_API Conv2dAlgorithm {
        public:
           static void conv3x3_winograd23_transform_kernel(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd23_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd63_transform_kernel(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd63_transform_kernel_inplace(const Tensor& kernel, Tensor &kernel_tm);

           static void conv3x3_winograd23_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd23(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd63_threadpool(const Tensor &x, const Tensor &k_tm, Tensor &out);

           static void conv3x3_winograd63(const Tensor &x, const Tensor &w, Tensor &out);

           static void conv2d_3x3_sse(const Tensor &x, const Tensor &w, Tensor &out);

           static void conv2d_3x3_sse_inplace(const Tensor &x, const Tensor &w, Tensor &out);

           //pack
           static void kernel_pack8x8(const Tensor &kernel, Tensor& kernel_packed);

           //static void col_pack8x8(const Tensor& col_tensor, int col_h, int col_w, Tensor& col_packed);

           //static void gemm_pack8x8(int M, int N, int K, const Tensor& kernel_packed, const Tensor& col_packed, Tensor& out);

           static void col_pack8x8(const T* col_tensor, int col_h, int col_w, T* col_packed);

           static void gemm_pack8x8(int M, int N, int K, const T* kernel_packed, const T* col_packed, T* out);

           static void kernel_pack4x4(const Tensor &kernel, Tensor& kernel_packed);

           static void col_pack4x4(const T* col_tensor, int col_h, int col_w, T* col_packed);

           static void gemm_pack4x4(int M, int N, int K, const T* kernel_packed, const T* col_packed, T* out);
        };
    }

}

//extern template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT32>::declare>;
//extern template class ts::opt::Conv2dAlgorithm<ts::dtype<ts::FLOAT64>::declare>;
