//
// Created by kier on 2019/2/21.
//

#ifndef TENSORSTACK_BACKEND_BASE_CONV2D_TRANSPOSE_CORE_H
#define TENSORSTACK_BACKEND_BASE_CONV2D_TRANSPOSE_CORE_H

#include "backend/common_structure.h"
#include "core/tensor.h"
#include "runtime/stack.h"

namespace ts {
    namespace base {
        class Conv2DTransposeCore {
        public:
            virtual ~Conv2DTransposeCore() = default;

            /**
             * this is net backward for Conv2D
             * @param x input Tensor4D
             * @param padding
             * @param padding_value
             * @param w [input_channels, output_channels, height, width]
             * @param stride
             * @param dilation
             * @param format
             * @param out
             * @param stack
             * Notice the input_channels and output_channels swapped in weights, as it is the backward for Conv2D
             */
            virtual void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                          const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                          Conv2DFormat format, Tensor &out, Stack &stack) {
                TS_LOG_ERROR << "What a Terrible Failure: not implement transpose conv2d core." << eject;
            }

            virtual void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                          const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                          Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                this->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }
        };

        /**
         * use Core implement Conv2DTranspose
         * @tparam Core must be the subclass of Conv2DTransposeCore
         */
        template<typename Conv2DTranspose, typename Core>
        class Conv2DTransposeWithCore : public Conv2DTranspose {
        public:
            using self = Conv2DTransposeWithCore;

            Conv2DTransposeWithCore() {
                m_core = std::make_shared<Core>();
            }

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack) override {
                m_core->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                this->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

        private:
            std::shared_ptr<Core> m_core;
        };


        template<typename Conv2DTranspose, typename Core>
        class PackedConv2DTransposeWithCore : public Conv2DTranspose {
        public:
            using self = PackedConv2DTransposeWithCore;

            PackedConv2DTransposeWithCore() {
                m_core = std::make_shared<Core>();
            }

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override {
                m_core->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack,
                                         kernel_packed);
            }

            void conv2d_transpose(const Tensor &x, const Padding2D &padding, float padding_value,
                                  const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                  Conv2DFormat format, Tensor &out, Stack &stack) override {
                this->conv2d_transpose(x, padding, padding_value, w, stride, dilation, format, out, stack, false);
            }

        private:
            std::shared_ptr<Core> m_core;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_CONV2D_TRANSPOSE_CORE_H
