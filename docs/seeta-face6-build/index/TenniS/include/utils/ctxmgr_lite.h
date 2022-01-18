//
// Created by kier on 2019-04-14.
//

#ifndef TENSORSTACK_UTILS_CTXMGR_LITE_H
#define TENSORSTACK_UTILS_CTXMGR_LITE_H

#include <thread>

#include "utils/api.h"
#include "utils/except.h"

namespace ts {
    class TS_DEBUG_API NoLiteContextException : public Exception {
    public:
        NoLiteContextException();

        explicit NoLiteContextException(const std::thread::id &id);

        NoLiteContextException(const std::string &name);

        explicit NoLiteContextException(const std::string &name, const std::thread::id &id);

    private:
        std::thread::id m_thread_id;
    };

    namespace ctx {
        namespace lite {
            template <typename T>
            class TS_DEBUG_API of {
            public:
                static void set(T *ctx);
                static T *get();
                static T &ref();
            };

            template<typename T>
            class TS_DEBUG_API bind {
            public:
                using self = bind;

                explicit bind(T *ctx)
                    : m_pre_ctx(of<T>::get()) {
                    of<T>::set(ctx);
                }

                explicit bind(T &ctx_ref)
                        : bind(&ctx_ref) {
                }

                ~bind() {
                    of<T>::set(m_pre_ctx);
                }

                bind(const self &) = delete;

                self &operator=(const self &) = delete;

            private:
                T *m_pre_ctx = nullptr;
            };

            template <typename T>
            inline void set(T *ctx) {
                of<T>::set(ctx);
            }

            template <typename T>
            inline void set(T &ctx_ref) {
                of<T>::set(&ctx_ref);
            }

            template<typename T>
            inline T *get() {
                return of<T>::get();
            }

            template<typename T>
            inline T *ptr() {
                return of<T>::get();
            }

            template<typename T>
            inline T &ref() {
                return of<T>::ref();
            }
        }
        using namespace lite;
    }

    template <typename T>
    class TS_DEBUG_API SetupContext {
    public:
        using self = SetupContext;

        void setup_context() {
            ctx::of<T>::set(static_cast<T*>(this));
        }
    };

    TS_DEBUG_API std::string classname(const std::string &name);
}

#endif //TENSORSTACK_UTILS_CTXMGR_LITE_H
