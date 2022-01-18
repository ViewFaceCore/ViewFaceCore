//
// Created by kier on 19-4-24.
//

#include "seeta/FaceDetector.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main_image() {
    seeta::ModelSetting setting;
    setting.set_device(SEETA_DEVICE_CPU);
    setting.append("/home/kier/CLionProjects/FaceBox/torch-faceboxes/model.json");

    seeta::FaceDetector FD(setting);

    auto img = cv::imread("hu.ge.jpg");
    // auto img = cv::imread("1.jpg");
    std::cout << "Got image: [" << img.cols << ", " << img.rows << ", " << img.channels() << "]" << std::endl;

    SeetaImageData simg;
    simg.height = img.rows;
    simg.width = img.cols;
    simg.channels = img.channels();
    simg.data = img.data;

    auto faces = FD.detect(simg);

    std::cout << faces.size << std::endl;

    for (int i = 0; i < faces.size; ++i) {
        auto &face = faces.data[i];
        auto &pos = face.pos;

        cv::rectangle(img, cv::Rect(pos.x, pos.y, pos.width, pos.height), CV_RGB(0, 128, 128), 3);
    }

    cv::imshow("FaceBoxes", img);

    auto key = cv::waitKey();

    return 0;
}

int main_video() {
    seeta::ModelSetting setting;
    setting.set_device(SEETA_DEVICE_CPU);
    setting.append("/home/kier/CLionProjects/FaceBox/torch-faceboxes/model.json");

    seeta::FaceDetector FD(setting);

    cv::VideoCapture capture("me.mp4");
    cv::Mat img;

    while (capture.isOpened()) {
        capture.grab();
        capture.retrieve(img);

        if (img.empty()) break;

        cv::resize(img, img, cv::Size(320, 240));

        SeetaImageData simg;
        simg.height = img.rows;
        simg.width = img.cols;
        simg.channels = img.channels();
        simg.data = img.data;

        auto faces = FD.detect(simg);

        std::cout << faces.size << std::endl;

        for (int i = 0; i < faces.size; ++i) {
            auto &face = faces.data[i];
            auto &pos = face.pos;

            cv::rectangle(img, cv::Rect(pos.x, pos.y, pos.width, pos.height), CV_RGB(0, 128, 128), 3);
        }

        cv::imshow("FaceBoxes", img);

        auto key = cv::waitKey(30);
        if (key >= 0) break;
    }

    return 0;
}

int main() {
    return main_video();
}
