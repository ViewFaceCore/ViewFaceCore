//
// Created by kier on 2018/11/11.
//

#ifndef TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H
#define TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H

#include "ctxmgr_lite.h"

#if defined(_MSC_VER) && _MSC_VER < 1900 // lower then VS2015
#    define TS_LITE_THREAD_LOCAL __declspec(thread)
#else
#    define TS_LITE_THREAD_LOCAL thread_local
#endif

namespace ts {

    template<typename T>
    class __lite_context {
    public:
        using self = __lite_context;
        using context = void *;

        explicit __lite_context(context ctx);

        ~__lite_context();

        static void set(context ctx);

        static context get();

        static context try_get();

        __lite_context(const self &) = delete;

        self &operator=(const self &) = delete;

        context ctx() const;

    private:
        context m_pre_ctx = nullptr;
        context m_now_ctx = nullptr;
    };

    // move from header to source
    template<typename T>
    class __thread_local_lite_context {
    public:
        using self = __thread_local_lite_context;

        using context = void *;

        static context swap(context ctx);

        static void set(context ctx);

        static const context get();

        static const context try_get();

    private:
        static TS_LITE_THREAD_LOCAL context m_ctx;
    };

    template<typename T>
    TS_LITE_THREAD_LOCAL
    typename __thread_local_lite_context<T>::context
            __thread_local_lite_context<T>::m_ctx = nullptr;

    template<typename T>
    typename __thread_local_lite_context<T>::context
    __thread_local_lite_context<T>::swap(typename __thread_local_lite_context<T>::context ctx) {
        auto pre_ctx = m_ctx;
        m_ctx = ctx;
        return pre_ctx;
    }

    template<typename T>
    void __thread_local_lite_context<T>::set(typename __thread_local_lite_context<T>::context ctx) {
        m_ctx = ctx;
    }

    template<typename T>
    typename __thread_local_lite_context<T>::context const __thread_local_lite_context<T>::get() {
        if (m_ctx == nullptr) throw NoLiteContextException(typeid(T).name());
        return m_ctx;
    }

    template<typename T>
    typename __thread_local_lite_context<T>::context const __thread_local_lite_context<T>::try_get() {
        return m_ctx;
    }

    template<typename T>
    __lite_context<T>::__lite_context(typename __lite_context<T>::context ctx) {
        this->m_now_ctx = ctx;
        this->m_pre_ctx = __thread_local_lite_context<T>::swap(ctx);
    }

    template<typename T>
    __lite_context<T>::~__lite_context() {
        __thread_local_lite_context<T>::set(this->m_pre_ctx);
    }

    template<typename T>
    void __lite_context<T>::set(typename __lite_context<T>::context ctx) {
        __thread_local_lite_context<T>::set(ctx);
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::get() {
        return __thread_local_lite_context<T>::get();
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::try_get() {
        return __thread_local_lite_context<T>::try_get();
    }

    template<typename T>
    typename __lite_context<T>::context __lite_context<T>::ctx() const {
        return m_now_ctx;
    }

    namespace ctx {
        namespace lite {
            template <typename T>
            void of<T>::set(T *ctx) {
                __lite_context<T>::set(ctx);
            }
            template <typename T>
            T *of<T>::get() {
                return reinterpret_cast<T *>(__lite_context<T>::try_get());
            }
            template <typename T>
            T &of<T>::ref() {
                return *reinterpret_cast<T *>(__lite_context<T>::get());
            }
        }
    }
}

#define TS_LITE_CONTEXT(T) \
    template class ts::__thread_local_lite_context<T>; \
    template class ts::__lite_context<T>; \
    template class ts::ctx::lite::of<T>;

#endif  // TENSORSTACK_UTILS_CTXMGR_LITE_SUPPORT_H