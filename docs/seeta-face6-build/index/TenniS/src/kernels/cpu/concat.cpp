#include <kernels/cpu/concat.h>
#include <core/tensor_builder.h>

#include "backend/name.h"
#include "core/memory.h"
#include "global/operator_factory.h"
#include "global/hard_converter.h"

namespace ts {
    namespace cpu {

        template<typename T>
        static void cpu_concat_run(const std::vector<Tensor> &x, int dim, Tensor &out) {
            // auto device_type = out.device().type();
            auto device_id = out.device().id();
            // first sync all memory to running_memory_device, then doo all inner deivce copy
            auto memcpy_handler = HardConverter::Query(out.device().type(), out.device().type());

            TS_AUTO_CHECK(memcpy_handler != nullptr);

            auto input_shape = x[0].sizes();
            ts::Tensor &output_tensor = out;
            auto output_shape = output_tensor.sizes();
            T *output_data = output_tensor.data<T>();

            int num_concats = 1;
            int input_concat_size = 1;
            int output_concat_axis = output_shape[dim];

            for (int i = 0; i < dim; i++) {
                num_concats *= input_shape[i];
            }

            for (int i = dim + 1; i < input_shape.size(); i++) {
                input_concat_size *= input_shape[i];
            }

            int offset_concat_axis = 0;


            for (size_t i = 0; i < x.size(); i++) {
                const T *input_data = x[i].data<T>();
                int input_concat_axis = x[i].sizes()[dim];
                for (int j = 0; j < num_concats; j++) {
                    memcpy_handler(
                            device_id,
                            output_data + (j * output_concat_axis + offset_concat_axis) * input_concat_size,
                            device_id,
                            input_data + j * input_concat_axis * input_concat_size,
                            input_concat_axis * input_concat_size * sizeof(T));
                }
                offset_concat_axis += input_concat_axis;
            }
        }

        void Concat::concat(const std::vector<Tensor> &x, int dim, Tensor &out) {
            class uint128_t {
            public:
                uint64_t h;
                uint64_t l;
            };
            auto type_bytes = out.proto().type_bytes();
            switch (type_bytes) {
                case 1:
                    cpu_concat_run<uint8_t>(x, dim, out);
                    break;
                case 2:
                    cpu_concat_run<uint16_t>(x, dim, out);
                    break;
                case 4:
                    cpu_concat_run<uint32_t>(x, dim, out);
                    break;
                case 8:
                    cpu_concat_run<uint64_t>(x, dim, out);
                    break;
                case 16:
                    cpu_concat_run<uint128_t>(x, dim, out);
                    break;
                default: {
                    auto dtype = out.dtype();
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Concat, ts::CPU, name::layer::concat())