#include "backend/torch/base_dcn_v2_forward.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "global/operator_factory.h"
#include "kernels/cpu/operator_on_cpu.h"

#include "dcn/dcn_v2.h"

namespace ts {
    namespace cpu {
    class DCNV2Forward : public OperatorOnCPU<base::DCNV2Forward> {
        public:
            void
            forward(const Tensor &x, const Tensor &w, const Tensor &b, const Tensor &offset, const Tensor &mask,
                    const Padding2D &padding, const Stride2D &stride, const Dilation2D &dilation, int deformable_groups,
                    Conv2DFormat format, Tensor &out) override {
                KSize2D ksize = Size2D(w.size(2), w.size(3));
                TS_AUTO_CHECK(padding.top == padding.bottom);
                TS_AUTO_CHECK(padding.left == padding.right);
                auto output = dcn_v2_cpu_forward(x, w, b, offset, mask, ksize.height, ksize.width, stride.height, stride.width, padding.top, padding.left, dilation.height, dilation.width, deformable_groups, &out);
                TS_AUTO_CHECK(output.data() == out.data());
            }
        };
    }
}


using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(DCNV2Forward, CPU, name::layer::dcn_v2_forward())