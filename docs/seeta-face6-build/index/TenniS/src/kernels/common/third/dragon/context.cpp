//
// Created by kier on 2019/9/6.
//

#include "kernels/common/third/dragon/context.h"
#include "kernels/common/third/dragon/workspace.h"

namespace ts {
    namespace dragon {
        ts::dragon::BaseContext::BaseContext(Workspace *ws) {
            m_computing_device = ws->workbench()->device().computing_device;
            m_memory_device = ws->workbench()->device().memory_device;
        }

        void BaseContext::set_stream_id(int id) {

        }
    }
}
