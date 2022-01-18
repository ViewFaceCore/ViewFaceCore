//
// Created by kier on 2019/7/26.
//

#include "backend/base/base_tile.h"

#include "backend/name.h"
#include "core/tensor_builder.h"

namespace ts {
    namespace base {
        Tile::Tile() {
            field(name::repeats, REQUIRED);
        }

        void Tile::init() {
            supper::init();

            m_repeats = tensor::array::to_int(get(name::repeats));

            auto valid = true;
            auto zeros = false;
            for (auto repeat : m_repeats) {
                if (repeat < 0) {
                    valid = false;
                    break;
                }
                if (repeat == 0) {
                    zeros = true;
                }
            }

            if (!valid) {
                TS_LOG_ERROR << "Can not repeats " << to_string(m_repeats) << eject;
            }

            m_zeros = zeros;
        }

        static Shape infer_shape(Shape &x, Shape &repeats) {
            if (x.size() == repeats.size()) {
            } else if (x.size() > repeats.size()) {
                do {
                    repeats.insert(repeats.begin(), 1);
                } while (x.size() > repeats.size());
            } else{
                do {
                    x.insert(x.begin(), 1);
                } while (x.size() < repeats.size());
            }
            Shape y(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                y[i] = x[i] * repeats[i];
            }
            return y;
        }

        int Tile::infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
            TS_AUTO_CHECK(stack.size() == 1);

            auto &x = stack[0];

            auto local_x_shape = x.sizes();
            auto local_repeats = m_repeats;

            auto output_shape = infer_shape(local_x_shape, local_repeats);

            output.resize(1);
            output[0] = Tensor::Prototype(x.dtype(), output_shape);

            return 1;
        }

        int Tile::run(ts::Stack &stack) {
            auto memory_device = running_memory_device();

            auto x = stack[0].view(memory_device);

            auto local_x_shape = x.sizes();
            auto local_repeats = m_repeats;

            auto output_shape = infer_shape(local_x_shape, local_repeats);

            auto &out = *stack.push(x.dtype(), output_shape, memory_device);

            if (m_zeros) return 1;

            x = x.reshape(local_x_shape);

            tile(x, local_repeats.std(), out);

            return 1;
        }
    }
}
