//
// Created by kier on 2019-06-10.
//

#include <module/module.h>

int main() {
    auto module = ts::Module::Load("yolov3-tiny.coco.tsm");
    ts::plot_graph(std::cout, module->outputs());
    module = ts::Module::Translate(module, ts::ComputingDevice(ts::CPU), "--float16");
    ts::plot_graph(std::cout, module->outputs());
}