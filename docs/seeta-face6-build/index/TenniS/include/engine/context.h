//
// Created by kier on 2018/5/19.
//

#ifndef TENSORSTACK_ENGINE_CONTEXT_H
#define TENSORSTACK_ENGINE_CONTEXT_H

#include <memory>

namespace ts {

    /**
     * Context for running device, work with Workbench
     * call by
     */
    class Context {
    public:
        using self = Context;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        /**
         * callback when context initialize
         */
        virtual void init() = 0;

        /**
         * callback when context finalize
         */
        virtual void exit() = 0;

        /**
         * callback when context resume
         */
        virtual void resume() = 0;

        /**
         * callback when context suspend
         */
        virtual void suspend() = 0;
    };

}


#endif //TENSORSTACK_ENGINE_CONTEXT_H
