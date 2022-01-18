//
// Created by kier on 2019/3/6.
//

#include "kernels/cpu/gatherv2.h"
#include "global/operator_factory.h"
#include "backend/name.h"

#include <numeric>

namespace ts {
    namespace cpu {

        void GatherV2::gather(const Tensor &x, const Tensor &indices, Tensor &out) {
            auto memcpy_handler = HardConverter::Query(out.device().type(), x.device().type());
            TS_AUTO_CHECK(memcpy_handler != nullptr);
            auto device_id = out.device().id();

            auto &x_shape = x.sizes();
            auto &i_shape = indices.sizes();

            int axis = i_shape[i_shape.size() - 1];

            auto number = std::accumulate(i_shape.begin(), i_shape.end() - 1, 1, std::multiplies<int>());
            auto width = std::accumulate(x_shape.begin() + axis, x_shape.end(), 1, std::multiplies<int>());

            HypeShape norm_input_shape(x_shape);

            auto bytes = x.proto().type_bytes();
            auto width_bytes = width * bytes;

            auto x_data = x.data<char>();
            auto out_data = out.data<char>();
            auto indices_data = indices.data<int32_t>();

            //std::vector<int32_t> tmpshape(axis, 0); 
            std::vector<int32_t> coordinate_shape(x_shape.size(),0);

            for (int i = 0; i < number; ++i) {
                for(int k=0; k<axis; k++) {
                    coordinate_shape[k] = indices_data[i * axis + k]; 
                } 
                
                //std::copy(tmpshape.begin(), tmpshape.end(),coordinate_shape.begin());  
                int index =  norm_input_shape.to_index(coordinate_shape);
                auto src_ptr = x_data + index * bytes;
                auto dst_ptr = out_data + i * width_bytes;
                //std::cout << "index:" << index << ",i:" << i << ",width:" << width_bytes << std::endl;
                memcpy_handler(device_id, dst_ptr, device_id, src_ptr,size_t(width_bytes));
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(GatherV2, CPU, name::layer::gatherv2())
