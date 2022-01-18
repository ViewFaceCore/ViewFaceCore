//
// Created by seeta on 2018/7/31.
//

#ifndef ORZ_TOOLS_FUNCTION_H
#define ORZ_TOOLS_FUNCTION_H

#include <functional>

namespace orz {

    /**
     * Not support std::function, so not using now
     * @tparam FUNC
     */
    template<typename FUNC>
    class Function;

    template<typename CLAZZ, typename Ret, typename... Args>
    class Function<Ret(CLAZZ::*)(Args...)> {
    public:
        using Type = Ret(CLAZZ::*)(Args...);

        static Ret call(Type func, CLAZZ *object, Args &&... args) {
            return (object->*func)(std::forward<Args>(args)...);
        }
    };

    template<typename CLAZZ, typename... Args>
    class Function<void (CLAZZ::*)(Args...)> {
    public:
        using Type = void (CLAZZ::*)(Args...);

        static void call(Type func, CLAZZ *object, Args &&... args) {
            (object->*func)(std::forward<Args>(args)...);
        }
    };

    template<typename CLAZZ, typename... Args>
    class Function<void (CLAZZ::*)(Args...) const> {
    public:
        using Type = void (CLAZZ::*)(Args...) const;

        static void call(Type func, CLAZZ *object, Args &&... args) {
            (object->*func)(std::forward<Args>(args)...);
        }
    };

    template<typename Ret, typename... Args>
    class Function<Ret(*)(Args...)> {
    public:
        using Type = Ret(*)(Args...);

        static Ret call(Type func, Args &&... args) {
            return (*func)(std::forward<Args>(args)...);
        }
    };

    template<typename... Args>
    class Function<void (*)(Args...)> {
    public:
        using Type = void (*)(Args...);

        static void call(Type func, Args &&... args) {
            (*func)(std::forward<Args>(args)...);
        }
    };

    /**
     * for calling function (with param 1 is ref) using pointer (at param 1)
     * @tparam Ret return type
     * @tparam CLAZZ the first param type
     * @tparam Args the params left
     */
    template<typename Ret, typename CLAZZ, typename... Args>
    class Function<Ret(*)(CLAZZ &, Args...)> {
    public:
        using Type = Ret(*)(CLAZZ &, Args...);

        static Ret call(Type func, CLAZZ *obj, Args &&... args) {
            return (*func)(*obj, std::forward<Args>(args)...);
        }

        static Ret call(Type func, CLAZZ &obj, Args &&... args) {
            return (*func)(obj, std::forward<Args>(args)...);
        }
    };

    /**
     * for calling function (with param 1 is ref) using pointer (at param 1)
     * @tparam CLAZZ
     * @tparam Args
     * @note function return void
     */
    template<typename CLAZZ, typename... Args>
    class Function<void (*)(CLAZZ &, Args...)> {
    public:
        using Type = void (*)(CLAZZ &, Args...);

        static void call(Type func, CLAZZ *obj, Args &&... args) {
            (*func)(*obj, std::forward<Args>(args)...);
        }

        static void call(Type func, CLAZZ &obj, Args &&... args) {
            (*func)(obj, std::forward<Args>(args)...);
        }
    };
}

#endif //ORZ_TOOLS_FUNCTION_H
