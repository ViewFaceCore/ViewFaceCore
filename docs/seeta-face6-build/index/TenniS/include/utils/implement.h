//
// Created by kier on 2019/1/8.
//

#ifndef TENSORSTACK_IMPLEMENT_H
#define TENSORSTACK_IMPLEMENT_H

#include <memory>

#include "api.h"

namespace ts {
    template <typename T>
    class TS_DEBUG_API Declare {
    public:
        using self = Declare;

        using Type = T;

        template <typename ...Args>
        explicit Declare(Args &&...args) : m_impl(new T(std::forward<Args>(args)...)) {}

        const T *operator->() const { return m_impl.get(); }

        T *operator->() { return m_impl.get(); }

        const T &operator*() const { return *m_impl; }

        T &operator*() { return *m_impl; }

        const T *get() const { return m_impl.get(); }

        T *get() { return m_impl.get(); }

    private:
        std::shared_ptr<T> m_impl;
    };
}

#endif //TENSORSTACK_IMPLEMENT_H
