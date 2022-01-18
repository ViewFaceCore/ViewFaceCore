//
// Created by kier on 19-4-24.
//

#include "seeta/PoseEstimator.h"
#include "seeta/Common/Struct.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <fstream>
#include <chrono>

#include <FaceDetector.h>




//faceboxes_2019-4-29_0.90.sta
int main_image()
{
    seeta::ModelSetting setting;
    setting.set_device( SEETA_DEVICE_CPU );
    //setting.append("/home/kier/CLionProjects/FaceBox/torch-faceboxes/model.json");

    setting.append( "PoseEstimation1.1.0.json" );
    //seeta::FaceDetector FD(setting);
    seeta::v6::PoseEstimator PE( setting );

    auto img = cv::imread( "1.jpg" );
    // auto img = cv::imread("1.jpg");
    std::cout << "Got image: [" << img.cols << ", " << img.rows << ", " << img.channels() << "]" << std::endl;

    SeetaImageData simg;
    simg.height = img.rows;
    simg.width = img.cols;
    simg.channels = img.channels();
    simg.data = img.data;


    SeetaRect info;
    std::ifstream faceInfo( "faceInfo.txt" );
    faceInfo >> info.x >> info.y >> info.width >> info.height;
    faceInfo.close();

    //info.x = 560;
    //info.y = 424;
    //info.width = 960 - info.x;
    //info.height = 1152 - info.y;


    std::cout << "Detect face at: (" << info.x << ", " << info.y << ", " << info.width << ", " << info.height << ")" << std::endl;
    std::cout << std::endl;

    // 0.3 姿态估计
    std::cout << "== Start test ==" << std::endl;
    float yaw, pitch, roll;
    yaw = pitch = roll = 0;
    int N = 1;
    std::cout << "Compute " << N << " times. " << std::endl;

    using namespace std::chrono;
    microseconds duration( 0 );
    for( int i = 0; i < N; ++i )
    {
        if( i % 10 == 0 ) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        PE.Estimate( simg, info, &yaw, &pitch, &roll );
        auto end = system_clock::now();
        duration += duration_cast<microseconds>( end - start );
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;

    // 0.4 获取结果
    std::cout << "== Plot result ==" << std::endl;
    std::cout << "Result: (yaw, pitch, roll) = (" << yaw << ", " << pitch << ", " << roll << ")" << std::endl;
    std::cout << std::endl;


    return 0;
}


int main_video()
{
    seeta::ModelSetting setting;
    setting.set_device( SEETA_DEVICE_CPU );
    setting.append( "PoseEstimation1.1.0.json" );
    seeta::v6::PoseEstimator PE( setting );
    
    
    seeta::ModelSetting setting2;
    setting2.set_device( SEETA_DEVICE_CPU );
    //setting2.append( "faceboxes_2019-4-29_0.90.sta" );
    setting2.append("/wqy/seeta_sdk/sdk/sdk6.0/FaceBoxes/example/bin/faceboxes_2019-4-29_0.90.sta");
    seeta::FaceDetector FD(setting2);
    //FD.set(seeta::FaceDetector::PROPERTY_THRESHOLD, 0.7);
    //FD.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 60);


       std::string title = "Pose Esimation";
        cv::namedWindow(title, cv::WINDOW_NORMAL);

        cv::VideoCapture vc(0);
        cv::Mat frame;
        cv::Mat canvas;

        float pre_roll = -180;
        float pre_pitch = -180;
        float pre_yaw = -180;

        while (vc.isOpened())
        {
                if (cv::waitKey(33) >= 0) break;

                vc >> frame;
                if (!frame.data) continue;
                cv::flip(frame, canvas, 1);

                // 1.1 ×¼±¸ÈËÁ³¼ì²âÊ¹ÓÃµÄÊý¾Ý
                //VIPLImageData image(frame.cols, frame.rows, frame.channels());

                SeetaImageData simg;
                simg.height = frame.rows;
                simg.width = frame.cols;
                simg.channels = frame.channels();
                simg.data = frame.data;

                //image.data = frame.data;

                // 1.2 ½øÐÐÈËÁ³¼ì²â
                auto infos = FD.detect(simg);

                int line_width = 4;
                //std::cout << "size:" << infos.size  << "width:" << simg.width << ",height:" << simg.height << ",channels:" << simg.channels << std::endl;
                std::ostringstream oss;
                for (int i=0; i<infos.size; i++)
                {
                        float scale = infos.data[i].pos.width / 300.0;
                        float line = 30;
                        cv::rectangle(canvas, cv::Rect(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y, 
                                      infos.data[i].pos.width, infos.data[i].pos.height), cv::Scalar(128, 0, 0), scale * line_width);
                        float yaw, pitch, roll;
                        // 2.1 ½øÐÐÍ·²¿×ËÌ¬¹À¼Æ£¬·Ö±ðÓÐÈý¸ö·½Ïò£¬ÒÔ½Ç¶È±íÊ¾
                        PE.Estimate(simg, infos.data[i].pos, &yaw, &pitch, &roll);

                        if (fabs(yaw - pre_yaw) < 1) yaw = pre_yaw;
                        if (fabs(pitch - pre_pitch) < 1) pitch = pre_pitch;
                        if (fabs(roll - pre_roll) < 1) roll = pre_roll;
                        pre_yaw = yaw;
                        pre_pitch = pitch;
                        pre_roll = roll;

                        //std::cout << "roll:" << roll << ",pitch:" << pitch << ",yaw:" << yaw << std::endl;
                        oss.str("");
                        oss << "roll:  " << (roll >= 0 ? " " : "") << roll;
                        cv::putText(canvas, oss.str(), cv::Point(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y - 10 * scale), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
                        oss.str("");
                        oss << "pitch: " << (pitch >= 0 ? " " : "") << pitch;
                        cv::putText(canvas, oss.str(), cv::Point(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y - 10 * scale - 1 * scale * line), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
                        oss.str("");
                        oss << "yaw:   " << (yaw >= 0 ? " " : "") << yaw;
                        cv::putText(canvas, oss.str(), cv::Point(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y - 10 * scale - 2 * scale * line), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
                }

                cv::imshow(title, canvas);
        }

        return 0;


}


int test_fd() {
    seeta::ModelSetting setting;
    setting.set_device(SEETA_DEVICE_CPU);
    //setting.append("/home/kier/CLionProjects/FaceBox/torch-faceboxes/model.json");

    setting.append("/wqy/seeta_sdk/sdk/sdk6.0/FaceBoxes/example/bin/faceboxes_2019-4-29_0.90.sta");
    seeta::FaceDetector FD(setting);

    auto img = cv::imread("1.jpg");
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


int main()
{
    //return test_fd();
    //return main_image();
    return main_video();
}
