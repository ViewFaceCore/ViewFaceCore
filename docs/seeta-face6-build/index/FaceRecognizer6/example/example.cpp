//
// Created by kier on 19-4-24.
//

#include "seeta/FaceRecognizer.h"
#include "Struct_cv.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

int main_image() {
    seeta::cv::ImageData image = cv::imread("1.png");
    seeta::Rect face(97, 54, 127, 170);
    std::vector<SeetaPointF> points = {
            {134.929, 126.889},
            {190.865, 120.054},
            {167.091, 158.991},
            {143.787, 186.269},
            {193.805, 181.186},

    };

    seeta::ModelSetting setting;
    setting.set_id(0);
    setting.append("/Users/seetadev/Documents/SDK/CLion/FaceRecognizer6/python/mask_arcface_2020-3-4.json");

    setting.set_device(SEETA_DEVICE_CPU);
    seeta::FaceRecognizer FR_cpu(setting);

    std::cout << "Got image: [" << image.width << ", " << image.height << ", " << image.channels << "]" << std::endl;

    auto patch = FR_cpu.CropFaceV2(image, points.data());
    cv::imwrite("patch.png", seeta::cv::ImageData(patch).toMat());

    std::shared_ptr<float> features_cpu(new float[FR_cpu.GetExtractFeatureSize()]);

    FR_cpu.ExtractCroppedFace(patch, features_cpu.get());

    auto &FR = FR_cpu;
    auto &features = features_cpu;


    int N = 100;
    std::cout << "Compute " << N << " times. " << std::endl;

    using namespace std::chrono;
    microseconds duration(0);
    for (int i = 0; i < N; ++i)
    {
        if (i % 10 == 0) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        FR.ExtractCroppedFace(patch, features.get());
        auto end = system_clock::now();
        duration += duration_cast<microseconds>(end - start);
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;


    for (int i = 0; i < FR.GetExtractFeatureSize(); ++i) {
        std::cout << features.get()[i] << " ";
    }
    std::cout << std::endl;

    cv::waitKey();

    return 0;
}

int main_video() {
    return 0;
}

int main() {
    return main_image();
}
