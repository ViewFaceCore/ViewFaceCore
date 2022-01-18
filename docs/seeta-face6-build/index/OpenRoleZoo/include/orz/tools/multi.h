//
// Created by seeta on 2018/7/30.
//

#ifndef ORZ_TOOLS_MULTI_H
#define ORZ_TOOLS_MULTI_H

#include <vector>
#include <memory>
#include <functional>

namespace orz {
    template<typename T>
    class Multi {
    public:
        using self = Multi;

        using Clazz = T;

        template<typename... Args>
        explicit Multi(size_t N, Args &&... args)
                : m_cores(N) {
            for (auto &core : this->m_cores) core = std::make_shared<T>(std::forward<Args>(args)...);
        }

        T &operator[](size_t _index) { return *this->m_cores[_index]; }

        const T &operator[](size_t _index) const { return *this->m_cores[_index]; }

        T &core(size_t _index) { return *this->m_cores[_index]; }

        const T &core(size_t _index) const { return *this->m_cores[_index]; }

        T &operator*() { return *this->m_cores[0]; }

        const T &operator*() const { return *this->m_cores[0]; }

        T *operator->() { return &*this->m_cores[0]; }

        const T *operator->() const { return &*this->m_cores[0]; }

        size_t size() const { return this->m_cores.size(); }

        /**
         * call function on each object
         * @tparam FUNC function type, should be Ret(T*, Args...) or Ret(T::*)(Args...)
         * @tparam Args the params of FUNC
         * @param func instance of FUNC
         * @param args instances of Args
         * @note FUNC can be Ret(T&, Args...)
         */
        template<typename FUNC, typename... Args>
        void each(FUNC func, Args &&... args) {
            for (auto &core : this->m_cores) std::bind(func, &*core, std::forward<Args>(args)...)();
        }

    private:
        using ptr = std::shared_ptr<T>;
        std::vector<ptr> m_cores;
    };
}

#endif //ORZ_TOOLS_MULTI_H
