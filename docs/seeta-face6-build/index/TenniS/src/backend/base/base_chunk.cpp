//
// Created by kier on 2019-04-15.
//

#include "backend/base/base_chunk.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

#include "utils/box.h"

namespace ts {
    namespace base {
        Chunk::Chunk() {
            field(name::chunks, REQUIRED);
            field(name::dim, OPTIONAL, tensor::from<int32_t>(-2));
        }

        void Chunk::init() {
            supper::init();

            m_chunks = tensor::to_int(get(name::chunks));
            m_dim = tensor::to_int(get(name::dim));

            TS_AUTO_CHECK(m_chunks > 0);
        }

        static int infer_return_dim(Stack &stack, int m_chunks, int m_dim, std::vector<Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            int input_dims = int(x.dims());
            int fixed_dim = m_dim >= 0 ? m_dim : input_dims + m_dim;

            if (fixed_dim < 0 || fixed_dim >= input_dims) {
                TS_LOG_ERROR << "Chunk dim must in [-"
                             << input_dims << ", "
                             << input_dims << ")" << eject;
            }

            int dim_size = x.size(fixed_dim);
            if (dim_size < m_chunks) {
                TS_LOG_ERROR << "Chunk size must greater " << m_chunks << eject;
            }

            auto bins = split_bins(0, dim_size, m_chunks);

            output.resize(m_chunks);

            for (int i = 0; i < m_chunks; ++i) {
                DTYPE proto_dtype = x.dtype();
                Shape proto_shape = x.sizes();
                proto_shape[fixed_dim] = bins[i].second - bins[i].first;
                output[i] = Tensor::Prototype(proto_dtype, proto_shape);
            }

            return fixed_dim;
        }

        int Chunk::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            infer_return_dim(stack, m_chunks, m_dim, output);
            return m_chunks;
        }

        int Chunk::run(Stack &stack) {
            std::vector<Tensor::Prototype> output_protos;
            int fixed_dim = infer_return_dim(stack, m_chunks, m_dim, output_protos);

            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            std::vector<Tensor> out;
            for (auto &proto : output_protos) {
                out.emplace_back(*stack.push(proto, memory_device));
            }

            chunk(x, m_chunks, fixed_dim, out);

            Tensor packed;
            packed.pack(out);

            stack.push(packed);

            return 1;
        }

    }
}
