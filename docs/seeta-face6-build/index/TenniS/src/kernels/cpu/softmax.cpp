#include <kernels/cpu/softmax.h>
#include <core/tensor_builder.h>
#include "backend/name.h"
#include "global/operator_factory.h"
#include <algorithm>
#include <math.h>

#include <kernels/common/simd.h>
#ifdef TS_USE_OPENMP
#include <kernels/common/openmp.h>
#endif

namespace ts {
	namespace cpu {
        template<typename T>
        void cpu_softmax_compute_run(const Tensor &x, int m_dim, Tensor &out) {
            auto &output_shape = out.sizes();

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            int body_num = output_shape[m_dim];

            if (body_num == 1) {
                T one(1);
                memset(output_data, out.device(), out.count() * out.proto().type_bytes(),
                       &one, Device(CPU), sizeof(T));
                return;
            }

            int head_num = 1;
            for (int i = 0; i < m_dim; i++) {
                head_num *= output_shape[i];
            }
            int tail_num = 1;
            for (int i = m_dim + 1; i < output_shape.size(); i++) {
                tail_num *= output_shape[i];
            }


            HypeShape hype({head_num, body_num, tail_num});

            // as NCW format
            for (int n = 0; n < head_num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int w = 0; w < tail_num; ++w) {
                    auto channel_index = hype.to_index(n, 0, w);
                    const T *input_channel_data = &input_data[channel_index];
                    T *output_channel_data = &output_data[channel_index];
                    const T *loop_in = input_channel_data;
                    T *loop_out = output_channel_data;
                    T sum = 0;
                    for (int i = 0; i < body_num; ++i) {
                        T data = T(std::exp(*loop_in));
                        sum += data;
                        *loop_out = data;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                    loop_in = input_channel_data;
                    loop_out = output_channel_data;
                    for (int i = 0; i < body_num; ++i) {
                        *loop_out /= sum;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                }
            }
        }

        template<typename T>
        void cpu_smooth_softmax_compute_run(const Tensor &x, int m_dim, Tensor &out) {
            auto &output_shape = out.sizes();

            auto input_data = x.data<T>();
            auto output_data = out.data<T>();

            int body_num = output_shape[m_dim];

            if (body_num == 1) {
                T one(1);
                memset(output_data, out.device(), out.count() * out.proto().type_bytes(),
                        &one, Device(CPU), sizeof(T));
                return;
            }

            int head_num = 1;
            for (int i = 0; i < m_dim; i++) {
                head_num *= output_shape[i];
            }
            int tail_num = 1;
            for (int i = m_dim + 1; i < output_shape.size(); i++) {
                tail_num *= output_shape[i];
            }


            HypeShape hype({head_num, body_num, tail_num});

            // as NCW format
            for (int n = 0; n < head_num; ++n) {
#ifdef TS_USE_OPENMP
#pragma omp parallel for num_threads(openmp_threads())
#endif
                for (int w = 0; w < tail_num; ++w) {
                    auto channel_index = hype.to_index(n, 0, w);
                    const T *input_channel_data = &input_data[channel_index];
                    T *output_channel_data = &output_data[channel_index];
                    const T *loop_in = input_channel_data;
                    T *loop_out = output_channel_data;
                    T max = *loop_in;
                    for (int i = 1; i < body_num; ++i) {
                        loop_in += tail_num;
                        T data = *loop_in;
                        if (data > max) max = data;
                    }
                    loop_in = input_channel_data;
                    T sum = 0;
                    for (int i = 0; i < body_num; ++i) {
                        T data = T(std::exp(*loop_in - max));
                        sum += data;
                        *loop_out = data;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                    loop_in = input_channel_data;
                    loop_out = output_channel_data;
                    for (int i = 0; i < body_num; ++i) {
                        *loop_out /= sum;

                        loop_in += tail_num;
                        loop_out += tail_num;
                    }
                }
            }
        }

		template<typename T>
		void cpu_softmax_compute_run(const Tensor &x, int m_dim, bool m_smooth, Tensor &out) {
		    if (m_smooth) {
		        return cpu_smooth_softmax_compute_run<T>(x, m_dim, out);
		    } else {
                return cpu_softmax_compute_run<T>(x, m_dim, out);
		    }
		}

//        template<>
//        void cpu_softmax_compute_run<float>(const Tensor &x, int m_dim, bool m_smooth, Tensor &out) {
//            auto output_shape = out.sizes();
//
//            int pre_num = 1;
//            for (int i = 0; i < m_dim; i++) {
//                pre_num *= output_shape[i];
//            }
//            int inner_num = 1;
//            for (int i = m_dim + 1; i < output_shape.size(); i++) {
//                inner_num *= output_shape[i];
//            }
//
//            int axis = output_shape[m_dim];
//
//            auto device_type = x.device();
//            const float *input_data = x.data<float>();
//            float *output_data = out.data<float>();
//
//            //memcpy(output_data, device_type, count * sizeof(float), input_data, device_type, count * sizeof(float));
//
//            int scale_data_size = out.count() / axis;
//            int denominator_data_size = scale_data_size;
//
//            Shape scale_shape;
//            scale_shape.resize(1);
//            scale_shape[0] = scale_data_size;
//            Tensor scale_tensor(Tensor::InFlow::HOST, out.dtype(), scale_shape);
//            float *scale_data = scale_tensor.data<float>();
//
//            Shape denominator_shape;
//            denominator_shape.resize(1);
//            denominator_shape[0] = denominator_data_size;
//            Tensor denominator_tensor(Tensor::InFlow::HOST, out.dtype(), denominator_shape);
//            float *denominator_data = denominator_tensor.data<float>();
//
//            for (int i = 0; i < pre_num; i++) {
//                std::memset(denominator_data, 0, scale_data_size * sizeof(float));
//                int pre_offset = i * axis * inner_num;
//                if (m_smooth) {
//                    //Caculate max value
//                    memcpy(scale_data, device_type, inner_num * sizeof(float), input_data + pre_offset,
//                        device_type, inner_num * sizeof(float));
//                    //std::memcpy(scale_data,input_data + i * axis * inner_num, inner_num * sizeof(float));
//                    for (int j = 0; j < axis; j++) {
//                        int post_offset = j * inner_num;
//                        for (int k = 0; k < inner_num - 3; k += 4) {
//                            float *scale_temp = &scale_data[k];
//                            float32x4 scale_data_x4(scale_temp);
//                            float32x4 input_data_x4(&input_data[pre_offset + post_offset + k]);
//                            scale_data_x4 = max_float32x4(scale_data_x4, input_data_x4);
//                            scale_data_x4.store(scale_temp);
//                        }
//                        for (int k = inner_num/4*4; k < inner_num; k++) {
//                            scale_data[k] = std::max(scale_data[k],
//                                input_data[pre_offset + post_offset + k]);
//                        }
//                    }
//                    //Caculate numerator and denominator
//                    for (int j = 0; j < axis; j++) {
//                        int post_offset = j * inner_num;
//                        for (int k = 0; k < inner_num; k++) {
//                            output_data[pre_offset + post_offset + k] = exp(
//                                input_data[pre_offset + post_offset + k] - scale_data[k]);
//                            denominator_data[k] += output_data[pre_offset + post_offset + k];
//                        }
//                    }
//                }
//                else {
//                    //Caculate numerator and denominator
//                    for (int j = 0; j < axis; j++) {
//                        int post_offset = j * inner_num;
//                        for (int k = 0; k < inner_num; k++) {
//                            output_data[pre_offset + post_offset + k] = exp(
//                                input_data[pre_offset + post_offset + k]);
//                            denominator_data[k] += output_data[pre_offset + post_offset + k];
//                        }
//                    }
//                }
//                //Caculte output
//                for (int j = 0; j < axis; j++) {
//                    int post_offset = j * inner_num;
//                    for (int k = 0; k < inner_num - 3; k += 4) {
//                        float *output_temp = &output_data[pre_offset + post_offset + k];
//                        float32x4 output_data_x4(output_temp);
//                        float32x4 denominator_data_x4(&denominator_data[k]);
//                        output_data_x4 = output_data_x4 / denominator_data_x4;
//                        output_data_x4.store(output_temp);
//                    }
//                    for (int k = inner_num/4*4; k < inner_num; k++)
//                    {
//                        output_data[pre_offset + post_offset + k] =
//                            output_data[pre_offset + post_offset + k] / denominator_data[k];
//                    }
//                }
//
//            }
//        }

		void Softmax::softmax(const Tensor &x, int dim, bool smooth, Tensor &out) {
			// Notice: the all tensor' memory device are CPU, as given in running_memory_device
			DTYPE dtype = out.dtype();
			switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_softmax_compute_run<TYPE>(x, dim, smooth, out); break; }
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
TS_REGISTER_OPERATOR(Softmax, ts::CPU, name::layer::softmax())
