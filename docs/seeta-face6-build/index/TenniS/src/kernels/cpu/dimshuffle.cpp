#include <kernels/cpu/dimshuffle.h>
#include <core/tensor_builder.h>
#include <set>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {
    namespace cpu {
        void Dimshuffle::dimshuffle(const Tensor &x, int dim, const std::vector<int> &shuffle, Tensor &out) {
            // auto device_type = out.device().type();
            auto device_id = out.device().id();
            // first sync all memory to running_memory_device, then doo all inner deivce copy
            auto memcpy_handler = HardConverter::Query(out.device().type(), out.device().type());

            TS_AUTO_CHECK(memcpy_handler != nullptr);

            const Shape &shape = x.sizes();
            // const Shape &reshape = out.sizes();

            size_t preoffset = 1;
            for (int i = 0; i < dim; i++) {
                preoffset *= shape[i];
            }

			size_t stride = 1;
            for (size_t i = dim; i < shape.size(); i++) {
                stride *= shape[i];
            }

			size_t backstride = 1;
            backstride = stride / size_t(shape[dim]);
			size_t newstride = backstride * shuffle.size();
            int type_len = type_bytes(x.dtype());
            auto ncpy = backstride * size_t(type_len);

            const char *psrc = x.data<char>();
            char *pdst = out.data<char>();
            for (size_t k = 0; k < preoffset; k++) {
                for (size_t i = 0; i < shuffle.size(); i++) {
                    memcpy_handler(
                            device_id, pdst + type_len * (k * newstride + i * backstride),
                            device_id, psrc + type_len * (k * stride + shuffle[i] * backstride), ncpy);
                }
            }
        }
    }
}

///////////////////////////////////////////
using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Dimshuffle, CPU, name::layer::dimshuffle())
