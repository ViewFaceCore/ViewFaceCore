#ifndef TENSORSTACK_BACKEND_BASE_CONV2D_QUANTIZED_CORE_H
#define TENSORSTACK_BACKEND_BASE_CONV2D_QUANTIZED_CORE_H

#include "backend/common_structure.h"
#include "core/tensor.h"
#include "runtime/stack.h"

namespace ts {
    namespace base {
        class Conv2DQuantizedCore {
        public:
            virtual ~Conv2DQuantizedCore() = default;

            virtual void conv2d(const Tensor &x, const Padding2D &padding, float padding_value,
                const Tensor &w, const Stride2D &stride, const Dilation2D &dilation,
                Conv2DFormat format, std::vector<float>dequantize_scale, Tensor &out, Stack &stack) = 0;
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
                Conv2DFormat format, std::vector<float>dequantize_scale,
                Tensor &out, Stack &stack) override {
                m_core->conv2d(x, padding, padding_value, w, stride, dilation, format, dequantize_scale, out, stack);
            }

        private:
            std::shared_ptr<Core> m_core;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_CONV2D_QUANTIZED_CORE_H
