//
// Created by kier on 2019-09-03.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"
#include "kernels/gpu/operator_on_gpu.h"
#include "backend/base/base_roi_align.h"
#include "kernels/cpu/dragon/roi_align_op.h"
#include "global/fp16_operator_factory.h"

/**
 * THis cpu implement was fork from Dragon, SeetaTech
 */

namespace ts {
    namespace gpu {

        class ROIAlign : public OperatorOnGPU<base::ROIAlign> {
        public:
            std::shared_ptr<dragon::ROIAlignOp<dragon::CUDAContext>> m_dragon;

            void init() override {
                dragon::Workspace ws;
                m_dragon = std::make_shared<dragon::ROIAlignOp<dragon::CUDAContext>>(this, &ws);
            }

            int run(Stack &stack) override {
                m_dragon->bind_outputs(1);

                m_dragon->bind_inputs(std::vector<Tensor>(stack.begin(), stack.end()));
                m_dragon->RunOnDevice();
                stack.push(Tensor::Pack(m_dragon->outputs()));
                m_dragon->clean();
                return 1;
            }


            std::vector<Tensor> roi_align(
                    const std::vector<Tensor> &inputs,
                    int pool_h, int pool_w, float spatial_scale, int sampling_ratio) override {
                TS_LOG_ERROR << "What a Terrible Failure!" << eject;
                return {};
            }
        };
    }
}

using namespace ts;
using namespace gpu;
TS_REGISTER_OPERATOR(ROIAlign, GPU, name::layer::roi_align())
#ifdef TS_USE_CUDA_FP16
TS_REGISTER_FP16_OPERATOR(ROIAlign, GPU, name::layer::roi_align())
#endif
