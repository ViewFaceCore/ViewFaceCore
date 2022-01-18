//
// Created by Lby on 2017/10/9.
//

#ifndef ORZ_MEM_NEED_H
#define ORZ_MEM_NEED_H

#include "orz/tools/void_bind.h"

namespace orz {

    class need {
    public:

        template<typename FUNC>
        need(FUNC func) {
            task = void_bind(func);
        }

        template<typename FUNC, typename... Args>
        need(FUNC func, Args &&... args) {
            task = void_bind(func, std::forward<Args>(args)...);
        }

        ~need() { task(); }

    private:
        need(const need &that) = delete;

        need &operator=(const need &that) = delete;

        VoidOperator task;
    };
}

#endif //ORZ_NEED_H
