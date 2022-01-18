//
// Created by kier on 2019/3/7.
//

#ifndef TENSORSTACK_BOARD_HOOK_H
#define TENSORSTACK_BOARD_HOOK_H

#include <functional>
#include "utils/ctxmgr_lite.h"

#include <utils/api.h>

namespace ts {
    class Stack;
    class Operator;

    class TS_DEBUG_API Hook : public SetupContext<Hook> {
    public:
        using self = Hook;
        struct StructAfterRun {
            const Stack *stack;
            const Operator *op;
        };
        struct StructBeforeRun {
            const Stack *stack;
            const Operator *op;
        };

        using CallbackAfterRun = std::function<void(const StructAfterRun &)>;
        using CallbackBeforeRun = std::function<void(const StructBeforeRun &)>;

        void after_run(const CallbackAfterRun &handler) {
            m_after_run = handler;
        }

        void before_run(const CallbackBeforeRun &handler) {
            m_before_run = handler;
        }

        void emit_after_run(const StructAfterRun &info) {
            if (m_after_run) m_after_run(info);
        }

        void emit_before_run(const StructBeforeRun &info) {
            if (m_before_run) m_before_run(info);
        }

    private:
        CallbackAfterRun m_after_run;
        CallbackBeforeRun m_before_run;
    };
}


#endif //TENSORSTACK_BOARD_HOOK_H
