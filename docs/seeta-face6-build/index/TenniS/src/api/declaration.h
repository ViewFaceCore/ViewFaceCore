//
// Created by kier on 2019/3/16.
//

#ifndef TENNIS_API_DECLARATION_H
#define TENNIS_API_DECLARATION_H

#include <memory>
#include "errno.h"
#include "utils/except.h"

#define DECLARE_API_TYPE(API_TYPE, TS_TYPE) \
struct API_TYPE { \
    using self = API_TYPE; \
    template <typename... Args> \
    explicit API_TYPE(Args &&...args) { \
        this->pointer = std::make_shared<TS_TYPE>(std::forward<Args>(args)...); \
    } \
    explicit API_TYPE(std::shared_ptr<TS_TYPE> pointer) : pointer(std::move(pointer)) {} \
    std::shared_ptr<TS_TYPE> pointer; \
    const TS_TYPE *operator->() const { return pointer.get(); } \
    TS_TYPE *operator->() { return pointer.get(); } \
    const TS_TYPE &operator*() const { return *pointer; } \
    TS_TYPE &operator*() { return *pointer; } \
    const TS_TYPE *get() const { return pointer.get(); } \
    TS_TYPE *get() { return pointer.get(); } \
};

#define ts_false 0
#define ts_true 1

#define TRY_HEAD \
ts::api::ClearLEM(); \
try {

#define RETURN_OR_CATCH(ret, cat) \
return ret; \
} catch (const Exception &e) { \
ts::api::SetLEM(e.what()); \
return cat; \
}

#define TRY_TAIL \
} catch (const Exception &) { \
}

#endif //TENNIS_API_DECLARATION_H
