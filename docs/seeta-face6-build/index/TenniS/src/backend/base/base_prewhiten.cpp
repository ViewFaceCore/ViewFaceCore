//
// Created by kier on 2019/2/20.
//

#include "backend/base/base_prewhiten.h"

namespace ts {
    namespace base {
        void PreWhiten::active(const Tensor &x, Tensor &out) {
            TS_AUTO_CHECK(x.dims() > 0);

            prewhiten(x, out);
        }
    }
}
