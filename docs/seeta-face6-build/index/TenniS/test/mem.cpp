//
// Created by lby on 2018/2/11.
//

#include <core/hard_memory.h>
#include <core/controller.h>
#include <iostream>
#include <global/memory_device.h>
#include <global/setup.h>

#include <map>
#include <unordered_map>

int main()
{
    ts::setup();

    ts::HardMemory mem(ts::MemoryDevice(ts::CPU, 0));
    try {
        mem.resize(10000000000000000000UL);
    } catch (const ts::Exception &e) {
        std::cout << e.what() << std::endl;
    }
    mem.resize(10);
    mem.data<int>()[0] = 10;
    std::cout << mem.data<int>()[0] << std::endl;
    mem.dispose();

    ts::DynamicMemoryController c({ts::CPU, 0});
    ts::Memory a = c.alloc(123);

    ts::Memory b(ts::MemoryDevice(ts::CPU, 0), 256);

    std::cout << a.size() << std::endl;
    std::cout << b.size() << std::endl;

    a.data<int>()[0] = 12;

    ts::memcpy(b, a, 123);

    std::cout << b.data<int>()[0] << std::endl;

    std::cout << ts::ComputingMemory::Query(ts::Device(ts::CPU, 0)) << std::endl;

    try {
        ts::ComputingMemory::Query(ts::Device("ARM", 0));
    } catch (const ts::NoMemoryDeviceException &e) {
        std::cout << e.what() << std::endl;
    }

#ifdef TS_USE_CUDA
    ts::Memory host_data1 = c.alloc(4);
    ts::Memory host_data2 = c.alloc(4);
    ts::DynamicMemoryController device_ctrl({ts::GPU, 0});
    ts::Memory device_data = device_ctrl.alloc(4);
    host_data1.data<int>()[0] = 233;
    ts::memcpy(device_data, host_data1);
    ts::memcpy(host_data2, device_data);
    std::cout << host_data2.data<int>()[0] << std::endl;
#endif

    return 0;
}