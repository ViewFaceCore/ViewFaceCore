#ifndef TENSORSTACK_BACKEND_BASE_BASE_NON_MAX_SUPPRESSION_V3_H
#define TENSORSTACK_BACKEND_BASE_BASE_NON_MAX_SUPPRESSION_V3_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Non_Max_Suppression_V3 : public OperatorOnDevice {
        public:
            using self = Non_Max_Suppression_V3;
            using supper = OperatorOnDevice;

            Non_Max_Suppression_V3();  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x input tensor
             * @param scores input tensor
             * @param out
             */
            virtual void non_max_suppression_v3(const Tensor &x, const Tensor &scores, Tensor &out) = 0;

        protected:
            int m_max_output_size;
            float m_iou_threshold;
            float m_score_threshold;
            std::string m_mode;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_NON_MAX_SUPPRESSION_V3_H
