//
// Created by kier on 2019/10/21.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_BROADCASTV2_H
#define TENSORSTACK_BACKEND_BASE_BASE_BROADCASTV2_H

#include "operator_on_device.h"

namespace ts {
    namespace base {
        class BroadcastV2 : public OperatorOnDevice {
        public:
            using self = BroadcastV2;
            using supper = OperatorOnDevice;

            BroadcastV2() = default;  // tell me the operator memory

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            /**
             *
             * @param x
             * @param rhs
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void broadcast(const Tensor &x, Tensor &out) = 0;

            /**
             *
             * @param x is bias on dim
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void broad_with_bias(const Tensor &x, Tensor &out, int dim);

            /**
             *
             * @param lhs not specific
             * @param x is scalar
             * @param out
             * @note all tensor's dtype is same, and all tensors' memory device are give in constructor
             */
            virtual void broadcast_with_scalar(const Tensor &x, Tensor &out);

            static bool broadcast(Shape &x, const Shape &shape);

            /**
             * return if is an scalar, also seen as count == 1
             * @param shape
             * @return
             */
            static bool is_scalar(const Shape &shape);

            /**
             * return if is bias on RHS
             * @param lhs_shape
             * @param rhs_shape
             * @param dim
             * @return
             */
            static bool is_bias(Shape &lhs_shape, Shape &rhs_shape, int &dim);
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_BROADCASTV2_H
