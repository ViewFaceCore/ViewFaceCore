//
// Created by seeta on 2018/6/28.
//

#include <runtime/stack.h>
#include <iostream>
#include <runtime/workbench.h>
#include <global/setup.h>

int main() {
    ts::setup();

    ts::MemoryDevice device(ts::CPU);

    ts::Stack ss(device);
    ss.push(ts::DTYPE::INT8, {1});
    std::cout << ss.size() << std::endl;
    ss.push(-1);
    ss.push(-1);
    ss.push(-1);
    ss.push(1);
    ss.pop(2);
    std::cout << ss.size() << std::endl;
    // ss.rebase(2);
    ss.push_base(0);
    std::cout << ss.size() << std::endl;
    ss.pop_base();
    std::cout << ss.size() << std::endl;


}