//
// Created by kier on 2019-04-13.
//

#ifndef TENSORSTACK_FRONTEND_DESC_H
#define TENSORSTACK_FRONTEND_DESC_H

/**
 * compressor firstly convert params to bubble,
 * then frontend use it build graph
 *      intime use it run operator in time
 */

#include <module/bubble.h>

namespace ts {
    namespace desc {
        enum class ResizeType : int32_t {
            LINEAR = 0,
            CUBIC = 1,
            NEAREST = 2,
        };

        TS_DEBUG_API Bubble resize2d(ResizeType type = ResizeType::LINEAR);

        TS_DEBUG_API Bubble add();

        TS_DEBUG_API Bubble sub();

        TS_DEBUG_API Bubble mul();

        TS_DEBUG_API Bubble div();

        TS_DEBUG_API Bubble transpose(const std::vector<int32_t> &permute);

        TS_DEBUG_API Bubble sigmoid();

        TS_DEBUG_API Bubble gather(int32_t axis);

        TS_DEBUG_API Bubble concat(int32_t dim);

        TS_DEBUG_API Bubble softmax(int32_t dim, bool smooth = true);

        TS_DEBUG_API Bubble pad(float padding_value = 0);

        TS_DEBUG_API Bubble cast(DTYPE dtype);

        TS_DEBUG_API Bubble affine_sample2d(int32_t dim = -1,
                                            float outer_value = 0,
                                            desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Bubble affine_on_sample2d(int32_t dim = -1,
                                               desc::ResizeType type = desc::ResizeType::LINEAR);

        TS_DEBUG_API Bubble matmul(bool transpose = false);

        TS_DEBUG_API Bubble broadcast();
    }
}

#endif //TENSORSTACK_FRONTEND_DESC_H
