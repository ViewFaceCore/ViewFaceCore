//
// Created by kier on 19-4-24.
//

#include "seeta/FaceDetector.h"
#include "seeta/FaceLandmarker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

int main_image() {
    seeta::ModelSetting setting;
    setting.set_device(SEETA_DEVICE_CPU);
    setting.append("/Users/seetadev/Documents/Files/models/PD/tsm/PDv5.pts81.json");

    seeta::FaceLandmarker FL(setting);

    auto img = cv::imread("1.png");
    std::cout << "Got image: [" << img.cols << ", " << img.rows << ", " << img.channels() << "]" << std::endl;
    seeta::Rect face(97, 54, 127, 170);

    SeetaImageData image;
    image.height = img.rows;
    image.width = img.cols;
    image.channels = img.channels();
    image.data = img.data;

    auto points = FL.mark(image, face);

    int N = 100;
    std::cout << "Compute " << N << " times. " << std::endl;

    using namespace std::chrono;
    microseconds duration(0);
    for (int i = 0; i < N; ++i)
    {
        if (i % 10 == 0) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        auto _ = FL.mark(image, face);
        auto end = system_clock::now();
        duration += duration_cast<microseconds>(end - start);
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;


    for (auto &point : points) {
        std::cout << "[" << point.x << ", " << point.y << "]" << std::endl;
        cv::circle(img, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
    }

    cv::imshow("Test", img);
    cv::waitKey();

    return 0;
}


static cv::Scalar blue(255, 0, 0);
static cv::Scalar green(0, 255, 0);
static cv::Scalar red(0, 0, 255);

void main_video()
{
        std::time_t now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char buf[32];
        strftime(buf, 32, "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
        std::cout << buf << std::endl;

        cv::VideoCapture capture;
        capture.open(0);

        //seeta::FaceDetector FD("model/VIPLFaceDetector5.1.0.dat");
        //seeta::FaceLandmarker PD("model/VIPLPointDetector5.0.pts19.mask.dat");


        seeta::ModelSetting setting;
        setting.set_device(SEETA_DEVICE_GPU);
        //setting.append("/home/kier/CLionProjects/FaceBox/torch-faceboxes/model.json");

        setting.append("/wqy/seeta_sdk/sdk/sdk6.0/FaceBoxes/example/bin/faceboxes_2019-4-29_0.90.sta");
        seeta::FaceDetector FD(setting);

        seeta::ModelSetting setting2;
        setting2.set_device(SEETA_DEVICE_GPU);
        setting2.append("./model/FL_mask_cnn6mb1_pts5.json");

        seeta::FaceLandmarker FL(setting2);


        //FD.SetMinFaceSize(120);
        //FD.SetScoreThresh(0.7, 0.7, 0.85);
        //FD.SetVideoStable(true);
        //PD.SetStable(true);


        cv::Mat mat;

        //std::vector<PointStable> ps;

        std::string title = "Point detector demo";
        //cv::namedWindow(title, CV_WINDOW_NORMAL);

         capture.set(CV_CAP_PROP_FRAME_WIDTH, 800);
         capture.set(CV_CAP_PROP_FRAME_HEIGHT, 600);

         //capture.set(CV_CAP_PROP_FRAME_WIDTH, 1920);
         //capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1080);

        int VIDEO_WIDTH = capture.get(CV_CAP_PROP_FRAME_WIDTH);
        int VIDEO_HEIGHT = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

        std::cout << "width:" << VIDEO_WIDTH << ", height:" << VIDEO_HEIGHT << std::endl;
        int landmark_num = FL.number();


        std::vector<SeetaPointF> landmarks;
        landmarks.resize(landmark_num);
        std::vector<int> masks;
        masks.resize(landmark_num);

        while (capture.isOpened())
        {
                capture >> mat;
                if (!mat.data) break;

                SeetaImageData simg;
                simg.height = mat.rows;
                simg.width = mat.cols;
                simg.channels = mat.channels();
                simg.data = mat.data;

                auto infos = FD.detect(simg);

                for (int n = 0; n < infos.size; ++n)
                {
                        auto &info = infos.data[n].pos;

                        FL.mark(simg, info, landmarks.data(), masks.data());

                        // cv::rectangle(mat, cv::Rect(info.x, info.y, info.width, info.height), blue, 3);
                        auto color = cv::Scalar(200, 200, 200);
                        color = cv::Scalar(0xFF, 0xFF, 0x97);
                        for (size_t i = 0; i < landmarks.size(); ++i)
                        {
                             auto &point = landmarks[i];
                             auto &mask = masks[i];
                             cv::circle(mat, cv::Point(point.x, point.y), 3, (mask ? red : green), -1);
                        }
                }

                cv::imshow(title, mat);

                if (cv::waitKey(1) >= 0) break;
        }
}

int main() {
    //return main_image();
    main_video();
    return 0;
}
