//
// Created by yang on 2019/10/9.
//

#include "benchmark/benchmark.h"
#include <utility>
#include <cctype>

int main(int argc, const char* argv[]){
    using namespace ts;

    if(argc < 4){
        std::cerr << "Usage: <command> loop_counts num_threads [device [id]] need_statistical power_mode compile_option" << std::endl;
        return 1;
    }

    int loop_count = int(std::strtol(argv[1], nullptr, 10));
    int num_thread = int(std::strtol(argv[2], nullptr, 10));
    std::string device = argv[3];
    for (auto &ch : device) {
        ch = char(std::tolower(ch));
    }

    int id = 0;
    if(argc > 4){
        id = int(std::strtol(argv[4], nullptr, 10));
    }

    bool need_statistical = false;
    if(argc > 5){
        std::string str = argv[5];
        for (auto &ch : str) {
            ch = char(std::tolower(ch));
        }
        need_statistical = str == "true" ? true : false;
    }

    int power_mode = -1;
    if(argc > 6){
        power_mode = int(std::strtol(argv[6], nullptr, 10));
    }

    std::string compile_option = "--pack";
    if(argc > 7){
        compile_option = argv[7];
        for (int i = 8; i < argc; ++i) {
            compile_option += " ";
            compile_option += argv[i];
        }
    }

    Option option;
    option.loop_counts = loop_count;
    option.num_threads = num_thread;
    option.device = device == "cpu" ? CPU : GPU;
    option.id = id;
    option.compile_option = compile_option;
    option.power_mode = power_mode;

    //print information
    std::cout << "loop_counts: " << option.loop_counts
              << " ,num_threads: " << option.num_threads
              << " ,device: " << option.device
              << " ,id: " << option.id
              << " ,need statistical: " << need_statistical
              << " ,compile option: " << option.compile_option
              << " ,power mode: " << option.power_mode << std::endl;

    BenchMark bm(option, need_statistical);

    bm.benchmark("facebox", "D:/yang/workPro/tensorStack/model_zoo/model/faceboxes_ipc_2019-7-12.tsm", {1, 3, 1280, 960});
    bm.benchmark("resnet50_ours", "D:/yang/workPro/tensorStack/model_zoo/model/RN50.tsm", {1, 3, 248, 248});
    bm.benchmark("resnet30_ours", "D:/yang/workPro/tensorStack/model_zoo/model/RN30.tsm", {1, 3, 248, 248});
    bm.benchmark("resnet14_ours", "D:/yang/workPro/tensorStack/model_zoo/model/antiface_model/_acc97_res14YuvAlignment_68000_0622.tsm", {1, 3, 256, 256});
    bm.benchmark("resnet18_ours", "D:/yang/workPro/tensorStack/model_zoo/model/antiface_model/_acc976_res182timeYuvAlignment_120000_190307.tsm", {1, 3, 256, 256});
    bm.benchmark("mobilenet_v2_ssd_ours", "D:/yang/workPro/tensorStack/model_zoo/model/antiface_model/SeetaAntiSpoofing.plg.1.0.m01d29.tsm", {1, 3, 300, 300});

    bm.benchmark("alexnet", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/alexnet-owt-4df8aa71.pth.tsm", {1, 3, 252, 252});
    bm.benchmark("densenet121", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/densenet121-a639ec97.pth.tsm", {1, 3, 224, 224});
    bm.benchmark("inception_v3", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/inception_v3_google-1a9a5a14.pth.tsm", {1, 3, 256, 256});
    bm.benchmark("mobilenet_v2", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/mobilenet_v2-b0353104.pth.tsm", {1, 3, 256, 256});
    bm.benchmark("resnet18", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/resnet18-5c106cde.pth.tsm", {1, 3, 256, 256});
    bm.benchmark("resnet50", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/resnet50-19c8e357.pth.tsm", {1, 3, 256, 256});
    bm.benchmark("squeezenet1_0", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/squeezenet1_0-a815701f.pth.tsm", {1, 3, 248, 248});
    bm.benchmark("squeezenet1_1", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/squeezenet1_1-f364aa15.pth.tsm", {1, 3, 256, 256});
    bm.benchmark("vgg16_bn", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/vgg16_bn-6c64b313.pth.tsm", {1, 3, 224, 224});
    bm.benchmark("vgg16", "D:/yang/workPro/tensorStack/model_zoo/model/third_party/pytorch/tsm/vgg16-397923af.pth.tsm", {1, 3, 224, 224});

    return 0;
}

