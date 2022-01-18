//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_CONV2D_CORE_H
#define TENSORSTACK_BACKEND_BASE_CONV2D_CORE_H

#include "backend/common_structure.h"
#include "core/tensor.h"
#include "runtime/stack.h"

namespace ts {
    namespace base {
        class Conv2DCore {
        public:
            virtual ~Conv2DCore() = default;

            virtual void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                conv2d(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

            virtual void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                Conv2DFormat format, Tensor &out, Stack &stack) {
                TS_LOG_ERROR << "What a Terrible Failure: not implement conv2d core." << eject;
            }
        };

        /**
         * use Core implement Conv2D
         * @tparam Core must be the subclass of Conv2DCore
         */
        template <typename Conv2D, typename Core>
        class Conv2DWithCore : public Conv2D {
        public:
            using self = Conv2DWithCore;

            Conv2DWithCore() {
                m_core = std::make_shared<Core>();
            }

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override {
                if (kernel_packed) {
                    TS_LOG_ERROR << "What a Terrible Failure: dealing packed weights without pack support." << eject;
                }
                conv2d(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack) override {
                m_core->conv2d(x, padding, padding_value, w, stride, dilation, format, out, stack);
            }

        private:
            std::shared_ptr<Core> m_core;
        };

        /**
         * use Core implement Conv2D
         * @tparam Core must be the subclass of Conv2DCore
         */
        template <typename Conv2D, typename Core>
        class PackedConv2DWithCore : public Conv2D {
        public:
            using self = PackedConv2DWithCore;

            PackedConv2DWithCore() {
                m_core = std::make_shared<Core>();
            }

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                        const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                        Conv2DFormat format, Tensor &out, Stack &stack, bool kernel_packed) override {
                m_core->conv2d(x, padding, padding_value, w, stride, dilation, format, out, stack, kernel_packed);
            }

            void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                                Conv2DFormat format, Tensor &out, Stack &stack) override {
                TS_LOG_ERROR << "What a Terrible Failure: not implement conv2d core." << eject;
            }

        private:
            std::shared_ptr<Core> m_core;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_CONV2D_CORE_H
