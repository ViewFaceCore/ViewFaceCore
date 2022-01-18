//
// Created by kier on 2019/2/16.
//

#ifndef TENSORSTACK_BACKEND_BASE_POOLING2D_CORE_H
#define TENSORSTACK_BACKEND_BASE_POOLING2D_CORE_H

#include "backend/common_structure.h"
#include "core/tensor.h"
#include "runtime/stack.h"

namespace ts {
    namespace base {
        class Pooling2DCore {
        public:
            virtual ~Pooling2DCore() = default;

            virtual void pooling2d(const Tensor &x, Pooling2DType type,
                                   const Padding2D &padding, Padding2DType padding_type,
                                   const Size2D &ksize, const Stride2D &stride,
                                   Conv2DFormat format, Tensor &out) = 0;
        };

        /**
         * use Core implement Pooling2D
         * @tparam Core must be the subclass of Pooling2DCore
         */
        template <typename Conv2D, typename Core>
        class Pooling2DWithCore : public Conv2D {
        public:
            using self = Pooling2DWithCore;

            Pooling2DWithCore() {
                m_core = std::make_shared<Core>();
            }

            void pooling2d(const Tensor &x, Pooling2DType type,
                           const Padding2D &padding, Padding2DType padding_type,
                           const Size2D &ksize, const Stride2D &stride,
                           Conv2DFormat format, Tensor &out) override {
                m_core->pooling2d(x, type, padding, padding_type, ksize, stride, format, out);
            }

        private:
            std::shared_ptr<Core> m_core;
        };
    }
}

#endif //TENSORSTACK_BACKEND_BASE_POOLING2D_CORE_H
