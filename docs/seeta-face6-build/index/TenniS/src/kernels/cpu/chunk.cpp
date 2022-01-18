#include <backend/base/base_chunk.h>
#include <core/tensor_builder.h>

#include "backend/name.h"
#include "core/memory.h"
#include "global/operator_factory.h"
#include "global/hard_converter.h"

namespace ts {
    namespace cpu {
        class Chunk : public OperatorOnAny<base::Chunk> {
        public:
            using self = Chunk;
            using supper = OperatorOnAny<base::Chunk>;

            Chunk() = default;

            void chunk(const Tensor &x, int chunks, int dim, std::vector<Tensor> &out) override;
        };

        template<typename T>
        static void cpu_chunk_run(const Tensor &x, int dim, std::vector<Tensor> &out) {
            // auto device_type = x.device().type();
            auto device_id = x.device().id();
            // first sync all memory to running_memory_device, then doo all inner deivce copy
            auto memcpy_handler = HardConverter::Query(x.device().type(), x.device().type());

            TS_AUTO_CHECK(memcpy_handler != nullptr);

            auto output_shape_template = out[0].sizes();
            const ts::Tensor &input_tensor = x;
            auto input_shape = input_tensor.sizes();
            const T *input_data = input_tensor.data<T>();
            int num_chunks = 1;
            int each_chunk_size = 1;
            int output_concat_axis = input_shape[dim];

            for (int i = 0; i < dim; i++) {
                num_chunks *= output_shape_template[i];
            }

            for (int i = dim + 1; i < output_shape_template.size(); i++) {
                each_chunk_size *= output_shape_template[i];
            }

            int offset_concat_axis = 0;


            for (size_t i = 0; i < out.size(); i++) {
                T *output_data = out[i].data<T>();
                int input_concat_axis = out[i].sizes()[dim];
                for (int j = 0; j < num_chunks; j++) {
                    memcpy_handler(
                            device_id,
                            output_data + j * input_concat_axis * each_chunk_size,
                            device_id,
                            input_data + (j * output_concat_axis + offset_concat_axis) * each_chunk_size,
                            input_concat_axis * each_chunk_size * sizeof(T));
                }
                offset_concat_axis += input_concat_axis;
            }
        }

        void Chunk::chunk(const Tensor &x, int chunks, int dim, std::vector<Tensor> &out) {
            class uint128_t {
            public:
                uint64_t h;
                uint64_t l;
            };
            auto type_bytes = x.proto().type_bytes();
            switch (type_bytes) {
                case 1:
                    cpu_chunk_run<uint8_t>(x, dim, out);
                    break;
                case 2:
                    cpu_chunk_run<uint16_t>(x, dim, out);
                    break;
                case 4:
                    cpu_chunk_run<uint32_t>(x, dim, out);
                    break;
                case 8:
                    cpu_chunk_run<uint64_t>(x, dim, out);
                    break;
                case 16:
                    cpu_chunk_run<uint128_t>(x, dim, out);
                    break;
                default: {
                    auto dtype = x.dtype();
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Chunk, ts::CPU, name::layer::chunk())