//
// Created by kier on 2019/3/6.
//

#include "kernels/cpu/gather.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {
        void Gather::gather(const Tensor &x, const Tensor &indices, int axis, Tensor &out) {
            auto memcpy_handler = HardConverter::Query(out.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy_handler != nullptr);
            auto device_id = out.device().id();

            auto &x_shape = x.sizes();
            auto number = std::accumulate(x_shape.begin(), x_shape.begin() + axis, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis + 1, x_shape.end(), 1, std::multiplies<int>());

            HypeShape norm_x_shape({number, x_shape[axis]});
            HypeShape norm_out_shape({number, indices.count()});

            auto bytes = x.proto().type_bytes();
            auto width_bytes = width * bytes;

            auto x_data = x.data<char>();
            auto out_data = out.data<char>();
            auto indices_data = indices.data<int32_t>();

            std::vector<int> norm_x_coord = {0, 0};
            std::vector<int> norm_out_coord {0, 0};

            for (int i = 0; i < norm_out_shape.shape(1); ++i) {
                norm_out_coord[1] = i;
                norm_x_coord[1] = indices_data[i];
                for (int n = 0; n < norm_out_shape.shape(0); ++n) {
                    norm_out_coord[0] = n;
                    norm_x_coord[0] = n;
                    auto src_ptr = x_data + norm_x_shape.to_index(norm_x_coord) * width_bytes;
                    auto dst_ptr = out_data + norm_out_shape.to_index(norm_out_coord) * width_bytes;
                    memcpy_handler(device_id, dst_ptr, device_id, src_ptr,size_t(width_bytes));
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Gather, CPU, name::layer::gather())
