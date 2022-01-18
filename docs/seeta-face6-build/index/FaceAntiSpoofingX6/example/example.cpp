//
// Created by kier on 19-4-24.
//

#include "seeta/PointDetector.h"
#include "seeta/FaceDetector.h"
#include "seeta/Common/Struct.h"

#include "seeta/Common/CStruct.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <fstream>
#include <chrono>

#include "seeta/FaceAntiSpoofing.h"

using namespace cv;

int main_video()
{
    bool isFirst = true;

    int device_id = 0;
    std::string ModelPath = "./model/";
    //FAS_model.set_device( device );
    //FAS_model.set_id( device_id );

    
    seeta::ModelSetting anti_setting;
    anti_setting.set_device( seeta::ModelSetting::CPU );
    anti_setting.append( "model/_acc976_res182timeYuvAlignment_120000_190307.json" );
    anti_setting.append( "model/SeetaAntiSpoofing.plg.1.0.m01d29.json" );
    //seeta::ModelSetting boxs_setting;
    //boxs_setting.set_device( seeta::ModelSetting::CPU );
    //boxs_setting.append( "model/SeetaAntiSpoofing.plg.1.0.m01d29.json" );

    //anti_setting.append( "model/SeetaAntiSpoofing.plg.1.0.m01d29.json" );
    //seeta::FaceAntiSpoofing processor( anti_setting, boxs_setting );
    seeta::FaceAntiSpoofing processor( anti_setting );
    processor.SetThreshold( 0.3, 0.80 );    // 设置默认阈值，另外一组阈值为(0.7, 0.55)
     

    seeta::ModelSetting FD_model;
    //FD_model.append( "/wqy/seeta_sdk/sdk/sdk6.0/FaceBoxes/example/bin/faceboxes_2019-4-29_0.90.sta" );
    FD_model.append( ModelPath + "VIPLFaceDetector5.1.1.sta" );
    FD_model.set_device( seeta::ModelSetting::CPU );
    FD_model.set_id( device_id );

    seeta::ModelSetting PD_model;
    PD_model.append( ModelPath + "SeetaPointDetector5.0.pts5.sta" );
    PD_model.set_device( seeta::ModelSetting::CPU );
    PD_model.set_id( device_id );


    seeta::FaceDetector fd( FD_model ); //人脸检测的初始化
    //FaceDetct.SetVideoStable( true );
    //FaceDetct.SetMinFaceSize( 100 );
    //FaceDetct.SetScoreThresh( 0.7, 0.7, 0.85 );
    //FaceDetct.SetImagePyramidScaleFactor( 1.414 );
    //fd.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 100);
    //fd.set(seeta::FaceDetector::PROPERTY_THRESHOLD, 0.7);

    seeta::PointDetector PointDetect( PD_model ); //关键点检测模型初始化


    Mat frame;
    VideoCapture capture( 0 ); //打开视频文件

    if( !capture.isOpened() )       //检测是否正常打开:成功打开时，isOpened返回ture
        std::cout << "fail to open!" << std::endl;
    while( true )
    {
        if( !capture.read( frame ) )
        {
            std::cout << "can not read any frame" << std::endl;
            break;
        }
        flip( frame, frame, 1 ); //左右旋转摄像头，使电脑中图像和人的方向一致
        if( frame.channels() == 4 )     //如果为4通道则转为3通道的rgb图像
        {
            cv::cvtColor( frame, frame, CV_RGBA2BGR );
        }

        //seeta::cv::ImageData image = frame;

        SeetaImageData image;
        image.height = frame.rows;
        image.width = frame.cols;
        image.channels = frame.channels();
        image.data = frame.data;
        // 从外部进行人脸检测和特征点定位
        auto faces = fd.detect( image );
        std::cout << "faces.size:" << faces.size << std::endl;
        if( faces.size == 1 )
        {
            auto &face = faces.data[0].pos;

            cv::Scalar color;
            color = CV_RGB( 0, 255, 0 );
            
            std::vector<SeetaPointF> points( PointDetect.GetLandmarkNumber() );
            PointDetect.Detect( image, face, points.data() );

            //单帧的活体检测
            //基于视频的活体检测
            auto status = processor.Predict( image, face, points.data() );
            std::string stateOfTheFace;
            switch( status )
            {
                case seeta::FaceAntiSpoofing::SPOOF:
                    stateOfTheFace = "spoof";
                    color = CV_RGB( 255, 0, 0 );
                    break;
                case seeta::FaceAntiSpoofing::REAL:
                    stateOfTheFace = "real";
                    color = CV_RGB( 0, 255, 0 );
                    break;
                case seeta::FaceAntiSpoofing::FUZZY:
                    // stateOfTheFace = "fuzzy";
                    break;
                case seeta::FaceAntiSpoofing::DETECTING:
                    // stateOfTheFace = "detecting";
                    break;
            }
            float clarity;
            float reality;

            processor.GetPreFrameScore( &clarity, &reality );

            std::cout << "Clarity = " << clarity << ", Reality = " << reality << std::endl;

            cv::putText( frame, stateOfTheFace, cv::Point( face.x, face.y - 10 ), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2 );
            
            rectangle( frame, cv::Rect( face.x, face.y, face.width, face.height ), color, 2, 8, 0 );
        }
        else
        {
            for( int i = 0; i < faces.size; i++ )
            {
                auto face = faces.data[i].pos;
                rectangle( frame, cv::Rect( face.x, face.y, face.width, face.height ), cv::Scalar( 255, 0, 0 ), 2, 8, 0 ); //画人脸检测框
            }
            processor.ResetVideo();
        }

        cv::imshow( "SeetaFaceAntiSpoofing", frame );   //显示视频
        if( cv::waitKey( 1 ) == 27 )    //退出条件：1，按exc键;2，达到显示“通过或未通过“的帧数;
            break;
    }


}


//faceboxes_2019-4-29_0.90.sta
int main_image()
{
    std::cout << "== Load model ==" << std::endl;
    const double faceSpoofDetctionThreshold = 0.86; //»îÌå¼ì²â¶þ·ÖÀàãÐÖµx
    seeta::ModelSetting anti_setting;
    anti_setting.set_device( seeta::ModelSetting::CPU );
    anti_setting.append( "model/_acc976_res182timeYuvAlignment_120000_190307.json" );
    anti_setting.append( "model/SeetaAntiSpoofing.plg.1.0.m01d29.json" );
    //seeta::ModelSetting boxs_setting;
    //boxs_setting.set_device( seeta::ModelSetting::CPU );
    //boxs_setting.append( "model/SeetaAntiSpoofing.plg.1.0.m01d29.json" );
    seeta::FaceAntiSpoofing asp( anti_setting );
    std::cout << "Load model success." << std::endl;
    std::cout << std::endl;


    // 0.1 ¼ÓÔØ´ýÊ¶±ðÍ¼Æ¬
    std::cout << "== Load image ==" << std::endl;

    cv::Mat img = cv::imread( "1.png", cv::IMREAD_COLOR );
    //VIPLImageData image = vipl_convert(mat);

    SeetaImageData simg;
    simg.height = img.rows;
    simg.width = img.cols;
    simg.channels = img.channels();
    simg.data = img.data;
    std::cout << "Load image: " << simg.width << "x" << simg.height << std::endl;
    std::cout << std::endl;

    // 0.2 ¼ÓÔØ¼ì²âµÄÈËÁ³£¬ÔÚÓ¦ÓÃÖÐÐèÒª¶¯Ì¬¼ì²âºÍÌØÕ÷µã¶¨Î»
    std::cout << "== Load face infomation ==" << std::endl;

    SeetaPointF points[5];
    std::ifstream landmarks( "landmarks.txt" );
    std::cout << "Detect landmarks at: [" << std::endl;
    for( int i = 0; i < 5; ++i )
    {
        landmarks >> points[i].x >> points[i].y;
        std::cout << "(" << points[i].x << ", " << points[i].y << ")," << std::endl;
    }
    std::cout << "]" << std::endl;
    landmarks.close();

    std::cout << std::endl;
    // 0.2 »îÌå¼ì²â
    seeta::FaceAntiSpoofing::Status status;

    int N = 1;
    std::cout << "Compute " << N << " times. " << std::endl;
    SeetaRect a;
    a.x = 0;
    a.y = 0;
    a.width = 100;
    a.height = 100;
    using namespace std::chrono;
    microseconds duration( 0 );
    for( int i = 0; i < N; ++i )
    {
        if( i % 10 == 0 ) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        status = asp.Predict( simg, a, points );
        auto end = system_clock::now();
        duration += duration_cast<microseconds>( end - start );
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;

    // 0.5 »ñÈ¡½á¹û
    std::string stateOfTheFace;

    switch( status )
    {
        case seeta::FaceAntiSpoofing::SPOOF:
            stateOfTheFace = "spoof";
            break;
        case seeta::FaceAntiSpoofing::REAL:
            stateOfTheFace = "real";
            break;
        case seeta::FaceAntiSpoofing::FUZZY:
            stateOfTheFace = "fuzzy";
            break;
    }

    std::cout << "== Plot result ==" << std::endl;
    //std::cout << "Face confidence: " << asp.GetLog() << std::endl;
    std::cout << "Face result: ";
    std::cout << stateOfTheFace;
    std::cout << std::endl;


    return 0;
}


int main()
{
    //return main_image();
    return main_video();
}
