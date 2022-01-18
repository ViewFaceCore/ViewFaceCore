//
// Created by kier on 19-4-24.
//

#include "seeta/EyeStateDetector.h"
#include "seeta/FaceDetector.h"
//#include "seeta/FaceLandmarker.h"
#include "seeta/Common/Struct.h"

//#include <VIPLPointDetector.h>
#include <seeta/PointDetector.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>

#include <fstream>


std::string getstatus(seeta::EyeStateDetector::EYE_STATE state)
{
    if(state == seeta::EyeStateDetector::EYE_CLOSE)
        return "CLOSE";
    else if (state == seeta::EyeStateDetector::EYE_OPEN)
        return "OPEN";
    else if (state == seeta::EyeStateDetector::EYE_RANDOM)
        return "RANDOM";
    else
        return "";
}

int main_image()
{
    seeta::ModelSetting setting;
    setting.set_device( SEETA_DEVICE_GPU );
    setting.set_id( 0 );
    setting.append( "SeetaEyeStateDetector1.3.99x99.json" );


    std::cout << "== Load model ==" << std::endl;

    seeta::EyeStateDetector EBD( setting );

    std::cout << "Load model success." << std::endl;
    std::cout << std::endl;

    // 0.1 ¼ÓÔØ´ýÊ¶±ðÍ¼Æ¬
    std::cout << "== Load image ==" << std::endl;

    cv::Mat mat = cv::imread( "1.png", cv::IMREAD_COLOR );
    //VIPLImageData image = vipl_convert(mat);

    SeetaImageData simage;
    simage.width = mat.cols;
    simage.height = mat.rows;
    simage.channels = mat.channels();
    simage.data = mat.data;


    std::cout << "Load image: " << mat.cols << "x" << mat.rows << std::endl;
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

    int status = 0;

    int N = 1;
    std::cout << "Compute " << N << " times. " << std::endl;

    seeta::EyeStateDetector::EYE_STATE leftstate,rightstate;
    using namespace std::chrono;
    microseconds duration( 0 );
    for( int i = 0; i < N; ++i )
    {
        if( i % 10 == 0 ) std::cout << '.' << std::flush;
        auto start = system_clock::now();
        EBD.Detect( simage, points, leftstate, rightstate );
        auto end = system_clock::now();
        duration += duration_cast<microseconds>( end - start );
    }
    std::cout << std::endl;
    double spent = 1.0 * duration.count() / 1000 / N;

    std::cout << "Average takes " << spent << " ms " << std::endl;
    std::cout << std::endl;

    // 0.5 »ñÈ¡½á¹û
    std::cout << "== Plot result ==" << std::endl;
    std::cout << "Eye status: (left, right) = " << "(";
    std::cout << getstatus(leftstate);
    std::cout << ", ";
    std::cout << getstatus(rightstate);
    std::cout << ")";
    std::cout << std::endl;

    return 0;

}

auto red = CV_RGB(255, 0, 0);
auto green = CV_RGB(0, 255, 0);
auto blue = CV_RGB(0, 0, 255);

class triger
{
public:
        bool signal(bool level)
        {
                bool now_level = !m_pre_level && level;
                m_pre_level = level;
                return now_level;
        }
private:
        bool m_pre_level = false;
};

/*
VIPLImageData vipl_convert(const cv::Mat &img)
{
        VIPLImageData vimg(img.cols, img.rows, img.channels());
        vimg.data = img.data;
        return vimg;
}
*/

int main_video()
{
    std::cout << "== Load model ==" << std::endl;

    seeta::ModelSetting setting;
    setting.set_device( SEETA_DEVICE_GPU );
    setting.set_id( 0 );
    setting.append( "SeetaEyeStateDetector1.3.99x99.json" );

    seeta::EyeStateDetector EBD( setting);


    seeta::ModelSetting setting2;
    setting2.set_device( SEETA_DEVICE_GPU );
    setting2.set_id( 0 );
    setting2.append( "/wqy/seeta_sdk/sdk/sdk6.0/FaceBoxes/example/bin/faceboxes_2019-4-29_0.90.sta" );
    seeta::FaceDetector FD( setting2 );
    //FD.SetVideoStable( true );

    seeta::ModelSetting setting3;
    setting3.set_device( SEETA_DEVICE_GPU );
    setting3.set_id( 0 );
    //setting3.append( "./SeetaPointDetector5.0.pts5.sta" );

    setting3.append( "./SeetaPointDetector5.1.pts5.sta" );
    seeta::PointDetector PD(setting3);
    //VIPLPointDetector PD("/wqy/seeta_sdk/sdk/PointDetector5/test/VIPLPointDetector5.0.pts5.dat");

    std::cout << "Load model success." << std::endl;
    std::cout << std::endl;

    // 0.1 ¼ÓÔØ´ýÊ¶±ðÍ¼Æ¬
    std::cout << "== Open camera ==" << std::endl;

    // cv::VideoCapture capture("WIN_20180609_12_31_01_Pro.mp4");
    cv::VideoCapture capture( 0 );
    cv::Mat frame, canvas;
    std::stringstream oss;

    triger triger_left, triger_right;
    int count_blink_times = 0;

    ///////////////////////////////////
    //cv::Mat mat = cv::imread( "1.png", cv::IMREAD_COLOR );
    //frame = mat.clone();

    /////////////////////////////////////

    while( capture.isOpened() )
    {
        capture >> frame;
        if( frame.empty() ) continue;
        canvas = frame.clone();


        SeetaImageData simage;
        simage.width = frame.cols;
        simage.height = frame.rows;
        simage.channels = frame.channels();
        simage.data = frame.data;
        //auto vframe = vipl_convert( frame );

        //std::cout << "begin FD " << std::endl;
        auto faces = FD.detect( simage );

        //std::cout << "FD ok" << std::endl;
        for( int i=0; i< faces.size; i++ )
        {

            SeetaPointF points[5];
            //VIPLPoint points[5];

            //VIPLFaceInfo info;
            //info.x = faces.data[i].x;
            //info.y = faces.data[i].y;
            //into.width = faces.data[i].width;
            //info.height = faces.data[i].height;

            //std::cout << "begin PD " << std::endl;
            PD.Detect( simage, faces.data[i].pos, points );
            //PD.DetectLandmarks( vframe, info, points );

            //std::cout << "PD ok" << std::endl;
            
            SeetaPointF pts[5];
            for(int m=0; m<5; m++) {
                pts[m].x = points[m].x;
                pts[m].y = points[m].y;
                //std::cout << "x:" << pts[m].x << ",y:" << pts[m].y << std::endl;
            }
           
/*        
                        double eyeSpan= sqrt(pow((points[1].x - points[0].x), 2) + pow((points[1].y - points[0].y), 2));

                        int leftEyePointX = std::max(int(points[0].x - eyeSpan / 2), 0);
                        int leftEyePointY = std::max(int(points[0].y - eyeSpan / 2), 0);
                        leftEyePointX = std::min(int(frame.cols - eyeSpan / 2), leftEyePointX);
                        leftEyePointY = std::min(int(frame.rows - eyeSpan / 2), leftEyePointY);
                        double leftEyeSpanTemp = std::max(int(eyeSpan), 1);
                        int leftEyeSpanX = (leftEyePointX + leftEyeSpanTemp > frame.cols - 1) ? frame.cols - 1 - leftEyePointX : leftEyeSpanTemp;
                        int leftEyeSpanY = (leftEyePointY + leftEyeSpanTemp > frame.rows - 1) ? frame.rows - 1 - leftEyePointY : leftEyeSpanTemp;
                        leftEyeSpanTemp = std::min(int(leftEyeSpanX), int(leftEyeSpanY));

                        cv::Mat leftEyeRoi = frame(cv::Rect(leftEyePointX, leftEyePointY, leftEyeSpanTemp, leftEyeSpanTemp));
                        //cv::resize(leftEyeRoi, leftEyeRoi, cv::Size(inputSizeX, inputSizeY));

                        int rightEyePointX = std::max(int(points[1].x - eyeSpan / 2), 0);
                        int rightEyePointY = std::max(int(points[1].y - eyeSpan / 2), 0);
                        rightEyePointX = std::min(int(frame.cols - eyeSpan / 2), rightEyePointX);
                        rightEyePointY = std::min(int(frame.rows - eyeSpan / 2), rightEyePointY);
                        double rightEyeSpanTemp = std::max(int(eyeSpan), 1);
                        int rightEyeSpanX = (rightEyePointX + rightEyeSpanTemp > frame.cols - 1) ? frame.cols - 1 - rightEyePointX : rightEyeSpanTemp;
                        int rightEyeSpanY = (rightEyePointY + rightEyeSpanTemp > frame.rows - 1) ? frame.rows - 1 - rightEyePointY : rightEyeSpanTemp;
                        rightEyeSpanTemp = std::min(int(rightEyeSpanX), int(rightEyeSpanY));

                        cv::Mat rightEyeRoi = frame(cv::Rect(rightEyePointX, rightEyePointY, rightEyeSpanTemp, rightEyeSpanTemp));
                        //resize(rightEyeRoi, rightEyeRoi, cv::Size(inputSizeX, inputSizeY));



                        std::cout << "++++x:" << leftEyePointX << ", y:" << leftEyePointY << ",width:" << leftEyeSpanTemp;
                        std::cout << "  x:" << rightEyePointX << ", y:" << rightEyePointY << ",width:" << rightEyeSpanTemp << std::endl;
            */ 
            //std::cout << "beign eye state " << std::endl;
            seeta::EyeStateDetector::EYE_STATE leftstate,rightstate;
            EBD.Detect( simage, pts, leftstate, rightstate );
            //bool blink = triger_left.signal( ( status & seeta::EyeBlinkDetector::LEFT_EYE ) && ( status & seeta::EyeBlinkDetector::RIGHT_EYE ) );
            // bool blink = triger_left.signal(EBD.ClosedEyes(vframe, points));

            oss.str( "left,right " );
            oss << "(";
            oss << getstatus(leftstate);
            oss << ", ";
            oss << getstatus(rightstate);
            oss << ")";

            cv::rectangle( canvas, cv::Rect( faces.data[i].pos.x, faces.data[i].pos.y, faces.data[i].pos.width, faces.data[i].pos.height ), cv::Scalar( 128, 0, 0 ), 3 );
            cv::putText( canvas, oss.str(), cv::Point( faces.data[i].pos.x, faces.data[i].pos.y - 10 ), 0, 0.5, cv::Scalar( 0, 128, 0 ), 2 );
        }



        cv::imshow( "Faces", canvas );
        auto key = cv::waitKey( 30 );
        if( key >= 0 ) break;
    }

    return EXIT_SUCCESS;


    return 0;
}

int main()
{
    return main_video();
    //return main_image();
}
