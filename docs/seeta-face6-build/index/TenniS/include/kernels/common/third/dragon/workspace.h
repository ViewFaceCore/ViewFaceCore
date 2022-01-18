//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_WORKWPACE_H
#define TENSORSTACK_THIRD_DRAGON_WORKWPACE_H

#include "runtime/workbench.h"
#include "type_meta.h"

namespace ts {
    namespace dragon {
        template<typename T>
        class Caches {
        public:
            Caches(const ts::Tensor &binary, const std::vector<int64_t> &segments)
                    : m_binary(binary) {
                m_shift.resize(segments.size());
                if (!m_shift.empty()) m_shift[0] = 0;
                for (size_t i = 1; i < m_shift.size(); ++i) {
                    m_shift[i] = m_shift[i - 1] + segments[i - 1];
                }
            }

            size_t size() const { return m_shift.size(); }

            T *operator[](size_t i) { return m_binary.data<T>() + m_shift[i]; }

            const T *operator[](size_t i) const { return m_binary.data<T>() + m_shift[i]; }

            T *operator[](int i) { return m_binary.data<T>() + m_shift[i]; }

            const T *operator[](int i) const { return m_binary.data<T>() + m_shift[i]; }

        private:
            std::vector<int64_t> m_shift;
            ts::Tensor m_binary;
        };

        class Workspace {
        public:
            ts::Workbench *workbench() {
                return &ctx::of<ts::Workbench>::ref();
            }

            template<typename T, typename Context>
            Caches<T> caches(const std::vector<int64_t> &segments) {
                auto N = std::accumulate(segments.begin(), segments.end(), int64_t(0));
                ts::Tensor binary;
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    binary = ts::Tensor(ts::Tensor::InFlow::HOST, dtypeid<T>::id, {int(N)});
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    binary = ts::Tensor(ts::Tensor::InFlow::DEVICE, dtypeid<T>::id, {int(N)});
                }
                return Caches<T>(binary, segments);
            }
        };
    }
}

#endif //TENSORSTACK_THIRD_DRAGON_WORKWPACE_H
