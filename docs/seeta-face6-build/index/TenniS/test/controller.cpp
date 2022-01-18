//
// Created by kier on 2019/1/8.
//

#include <memory/flow.h>

#include <utils/log.h>

int main() {
    using namespace ts;
    MemoryDevice device(CPU, 0);

    GlobalLogLevel(LOG_DEBUG);

    VatMemoryController controller(device);

    auto mem = controller.alloc(100);

    controller.alloc(12);
    controller.alloc(3);

    mem.data<int>()[0] = 3;

}

