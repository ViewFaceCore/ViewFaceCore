#include "kernels/cpu/topkv2.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>
#include <algorithm>

namespace ts {
    namespace cpu {
        template <typename T>
        static void adjust_node(T *arr, int n, int len, int *arr2) {
            int l, r, max, index;
            T tmp;
            l = 2 * n + 1; 
            r = 2 * n + 2;
            max = n;

            if (l<len&&arr[l]>arr[n])
                max = l;
            if (r<len&&arr[r]>arr[max])
                max = r;
    
            if (max != n) {
                tmp = arr[n];
                arr[n] = arr[max];
                arr[max] = tmp;

                index = arr2[n];
                arr2[n] = arr2[max];
                arr2[max] = index; 
                adjust_node(arr, max, len, arr2);
            }
        }

        template <typename T>
        static void sort_heap(T *arr, int len, int *arr2) {
            for (int i = len / 2; i >= 0; i--)
                adjust_node(arr, i, len, arr2);
            int index;
            T   tmp;
            for (int i = len - 1; i >= 0; i--) {
                tmp = arr[0];
                arr[0] = arr[i];
                arr[i] = tmp;

                index = arr2[0];
                arr2[0] = arr2[i];
                arr2[i] = index;
                adjust_node(arr, 0, i, arr2);
            }
        }

        template <typename T>
        static void cpu_topkv2_sorted_compute_run(const Tensor &x, int K, Tensor &values, Tensor &indices) {
            auto N = std::accumulate(x.sizes().begin(), x.sizes().end() - 1, 1, std::multiplies<int32_t>());
            auto W = x.sizes().back();
            std::vector<int32_t> indices_temp(W);
            for (int n = 0; n < N; ++n) {
                auto data = x.data<T>() + n * W;
                for (int w = 0; w < W; ++w) indices_temp[w] = w;
                std::partial_sort(indices_temp.begin(), indices_temp.begin() + K, indices_temp.end(),
                                  [data](int32_t a, int32_t b) { return data[a] > data[b]; });
                auto values_data = values.data<T>() + n * K;
                auto indices_data = indices.data<int32_t>() + n * K;
                std::memcpy(indices_data, indices_temp.data(), K * sizeof(int32_t));
                for (int k = 0; k < K; ++k) {
                    *values_data = data[indices_data[k]];
                    ++values_data;
                }
            }
        }


        template <typename T>
        static void cpu_topkv2_compute_run(const Tensor &x, int K, bool sorted, Tensor &values, Tensor &indices) {
            if (sorted) {
                cpu_topkv2_sorted_compute_run<T>(x, K, values, indices);
            } else {
                cpu_topkv2_sorted_compute_run<T>(x, K, values, indices);
            }
        }


        void Topkv2::topkv2(const Tensor &x, int K, bool sorted, Tensor &values, Tensor &indices) {
            DTYPE dtype = x.dtype();
           
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_topkv2_compute_run<TYPE>(x, K, sorted, values, indices); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }

        }

    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Topkv2, CPU, name::layer::topkv2())
