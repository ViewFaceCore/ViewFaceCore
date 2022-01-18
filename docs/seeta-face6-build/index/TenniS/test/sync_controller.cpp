//
// Created by kier on 2018/12/26.
//

#include "core/sync/sync_controller.h"
#include "core/controller.h"

int main() {
    auto controller = ts::HypeSyncMemoryController<ts::DynamicMemoryController>::Make(ts::MemoryDevice(ts::CPU), true);

    int i = 10;

    ts::MemoryDevice cpu0(ts::CPU, 0);
    ts::MemoryDevice cpu1(ts::CPU, 1);

    auto a = controller->alloc(cpu0, 4);

    TS_LOG_INFO << "Write i=" << i;
    a.view(cpu0).data<int>()[0] = i;

    TS_LOG_INFO << "Got i=" << a.view(cpu1).data<int>()[0];

    {
        TS_LOG_INFO << "Swap i=" << i;
        auto locked = a.view(cpu1);
        locked.data<int>()[0] = -i;
        locked.broadcast();
    }

    TS_LOG_INFO << "Got i=" << a.view(cpu0).data<int>()[0];
}

