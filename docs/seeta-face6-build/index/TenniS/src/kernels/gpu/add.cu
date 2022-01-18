#include <kernels/gpu/add.h>
#include <core/tensor_builder.h>
#include <backend/name.h>
#include <utils/assert.h>
#include <global/operator_factory.h>
#include <core/device.h>

#include <numeric>

#include <core/memory.h>

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "kernels/gpu/gpu_kernel.h"

#include "global/fp16_operator_factory.h"
#include <cuda_fp16.h>

namespace ts {
    namespace gpu {

        template<typename T>
        static __global__ void reduce_operator_scalar_kernel(T* data, int size, const T *scalar) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                data[index] += *scalar;
            }
        }

        template<typename T>
        static __global__ void reduce_operator_same_shape_kernel(T* data, const T*bias, int size) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                //int dim = index % ( step * slice ) / (step);
                data[index] += bias[index];
            }
        }

        template<typename T>
        static __global__ void reduce_operator_bias_kernel(T* data, int size, int step, int slice,
                                        const T* bias, int biaslen ) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index < size) {
                int dim = index % ( step * slice ) / (step);
                data[index] += bias[dim];
            }
        }


        template<typename T>
        static __global__ void reduce_operator_kernel(T* out, int size, const T* lhs,  const T* rhs, 
                                               int *lhsshape, int *lhsweight,  
                                               int *rhsshape, int *rhsweight, 
                                               int *outweight, int shapelen) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= size) 
                return;

            int *ptmp = outweight + 1;
            int ntmp = index;

            int rhsindex = 0;
            int lhsindex = 0;
            int nbuff1,nbuff2;
            nbuff1 = nbuff2 = 0;
            for(int m = 0, i = shapelen - 1; i >= 0; --i, m++) {
                if(i > 0) {
                    nbuff1 = ntmp / *ptmp;
                    ntmp %= *ptmp;
                }else {
                    nbuff1 = ntmp;
                }

                nbuff2 = nbuff1 % lhsshape[m];
                if(m < shapelen - 1) {
                    lhsindex += nbuff2 * lhsweight[m+1];
                }else {
                    lhsindex += nbuff2;
                }

                nbuff2 = nbuff1 % rhsshape[m];

                if(m < shapelen - 1) {
                    rhsindex += nbuff2 * rhsweight[m+1];
                }else {
                    rhsindex += nbuff2;
                }

                ++ptmp;
            }

            out[index] = lhs[lhsindex] + rhs[rhsindex];


        }


        template<typename T>
        static inline void add_gpu_compute_run(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            HypeShape lhs_hype(lhs.sizes());
            HypeShape rhs_hype(rhs.sizes());
            HypeShape out_hype(out.sizes());

            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto ncount = out.count();

            int *lhsshape = nullptr;
            int *rhsshape = nullptr;
            int *lhsweight = nullptr;
            int *rhsweight = nullptr;
            int *outweight = nullptr;

            /////////////////////////////////////
            Shape tmpshape;
            tmpshape.resize(1);
            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_tensor(out.device(), INT32, tmpshape);
            lhsshape = lhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_tensor(out.device(), INT32, tmpshape);
            rhsshape = rhs_tensor.data<int32_t>();

            tmpshape[0] = int32_t(lhs.sizes().size());
            Tensor lhs_weight_tensor(out.device(), INT32, tmpshape);
            lhsweight = lhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(rhs.sizes().size());
            Tensor rhs_weight_tensor(out.device(), INT32, tmpshape);
            rhsweight = rhs_weight_tensor.data<int32_t>();

            tmpshape[0] = int32_t(out.sizes().size());
            Tensor out_weight_tensor(out.device(), INT32, tmpshape);
            outweight = out_weight_tensor.data<int32_t>();

            memcpy((void*)lhsshape, out.device(), lhs.sizes().size() * sizeof(int32_t),
                   (void*)lhs.sizes().data(), MemoryDevice(CPU), lhs.sizes().size() * sizeof(int32_t));

            memcpy((void*)rhsshape, out.device(), rhs.sizes().size() * sizeof(int32_t),
                   (void*)rhs.sizes().data(), MemoryDevice(CPU), rhs.sizes().size() * sizeof(int32_t));

            memcpy((void*)lhsweight, out.device(), lhs_hype.weight().size() * sizeof(int32_t),
                   (void*)lhs_hype.weight().data(), MemoryDevice(CPU), lhs_hype.weight().size() * sizeof(int32_t));

            memcpy((void*)rhsweight, out.device(), rhs_hype.weight().size() * sizeof(int32_t),
                   (void*)rhs_hype.weight().data(), MemoryDevice(CPU), rhs_hype.weight().size() * sizeof(int32_t));
            memcpy((void*)outweight, out.device(), out_hype.weight().size() * sizeof(int32_t),
                   (void*)out_hype.weight().data(), MemoryDevice(CPU), out_hype.weight().size() * sizeof(int32_t));
            /////////////////////////////////////

            RUN_KERNEL(reduce_operator_kernel, CUDA_BLOCK(ncount, CUDA_THREAD_NUM), CUDA_THREAD_NUM, pout, ncount,
                        plhs, prhs, lhsshape, lhsweight, rhsshape, rhsweight, outweight, int(out.sizes().size()));
        }


        template<typename T>
        static inline void add_gpu_compute_run_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();
            
            memcpy((void*)pout, out.device(), out.count() * sizeof(T),
                   (void*)plhs, lhs.device(), out.count() * sizeof(T));

            RUN_KERNEL(reduce_operator_scalar_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, out.count(), prhs);
        }


        template<typename T>
        static inline void add_gpu_compute_run_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            memcpy((void*)pout, out.device(), out.count() * sizeof(T),
                   (void*)plhs, lhs.device(), out.count() * sizeof(T));

            RUN_KERNEL(reduce_operator_same_shape_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, prhs, out.count());
        }


        template<typename T>
        static inline void add_gpu_compute_run_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
            auto plhs = lhs.data<T>();
            auto prhs = rhs.data<T>();
            auto pout = out.data<T>();

            auto &out_shape = out.sizes();

            auto number = std::accumulate(out_shape.begin(), out_shape.begin() + dim, 1, std::multiplies<int>());
            auto count = std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

            auto channels = out_shape[dim];

            memcpy((void*)pout, out.device(), out.count() * sizeof(T),
                   (void*)plhs, lhs.device(), out.count() * sizeof(T));

            //RUN_KERNEL(reduce_operator_bias_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM, pout, out.count(), count, channels, prhs, rhs.count());
            RUN_KERNEL(reduce_operator_bias_kernel<T>, CUDA_BLOCK(out.count(), CUDA_THREAD_NUM), CUDA_THREAD_NUM,
                       pout, out.count(), count, channels, prhs, rhs.count());
        }


        void Add::reduce_with_broadcast(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { add_gpu_compute_run<TYPE>(lhs, rhs, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }

        void Add::reduce_with_scalar(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { add_gpu_compute_run_scalar<TYPE>(lhs, rhs, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }

        void Add::reduce_with_bias(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { add_gpu_compute_run_bias<TYPE>(lhs, rhs, out, dim); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }

        void Add::reduce_with_same_shape(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            // Notice: the all tensor' memory device are CPU, as given in running_memory_device
            DTYPE dtype = out.dtype();
            switch(dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { add_gpu_compute_run_same_shape<TYPE>(lhs, rhs, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
#ifdef TS_USE_CUDA_FP16
                DECLARE_COMPUTE_RUN(FLOAT16, half);
#endif
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }

        void Add::reduce_with_bias_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out, int dim) {
            this->reduce_with_bias(rhs, lhs, out, dim);
        }

        void Add::reduce_with_scalar_cross(const Tensor &lhs, const Tensor &rhs, Tensor &out) {
            this->reduce_with_scalar(rhs, lhs, out);
        }
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(Add, GPU, name::layer::add())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(Add, ts::GPU, name::layer::add())
#endif

