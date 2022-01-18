#include <iostream>

using namespace  std;

#include <seeta/FaceTracker.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

SeetaImageData seeta_convert( const cv::Mat &image )
{
    SeetaImageData simage;
    simage.width = image.cols;
    simage.height = image.rows;
    simage.channels = image.channels();
    simage.data = image.data;
    return simage;
}

int main( int argc, char *argv[] )
{

    cv::VideoCapture capture;   // 这里可以打开摄像头或视频
    if( argc < 2 )
    {
        std::cout << "Open Camera(0)" << std::endl;
        std::cout << "Use \"" << argv[0] << " video_file_name\" to open video." << std::endl;
        capture.open( 0 );
    }
    else
    {
        std::string video_file_name = argv[1];
        std::cout << "Open " << video_file_name << std::endl;
        capture.open( video_file_name );
    }

    // 获取输入流的长宽
    int video_width = static_cast<int>( capture.get( CV_CAP_PROP_FRAME_WIDTH ) );
    int video_height = static_cast<int>( capture.get( CV_CAP_PROP_FRAME_HEIGHT ) );


    // 注意这里要设置视频输入的大小
    seeta::FaceTracker tracker( seeta::ModelSetting( "model/VIPLFaceDetector5.1.1.sta" ), video_width, video_height );


    cv::Mat mat;
    int frameID = 0;

    while( capture.isOpened() )
    {
        capture >> mat;
        mat = mat.clone();
        if( !mat.data ) break;

        SeetaImageData simage = seeta_convert( mat );

        auto sinfos = tracker.Track( simage );
        std::vector<SeetaTrackingFaceInfo> result( sinfos.data, sinfos.data + sinfos.size );

        /*
         * VIPLTrackingInfo.PID 是跟踪赋予的 ID，相同的 ID 表示在视频内是同一个人
         * VIPLTrackingInfo.pos 是人脸的位置
         */
        for( auto &face : result )
        {
            cv::Scalar color = cv::Scalar( 255, 0, 0 );
            cv::rectangle( mat, cv::Rect( face.pos.x, face.pos.y, face.pos.width, face.pos.height ), color, 3 );
            std::ostringstream oss;
            oss << "PID: " << face.PID;
            cv::putText( mat, oss.str(), cv::Point( face.pos.x, face.pos.y - 5 ), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar( 0, 255, 3 ), 3 );
            std::cout << "PID(" << face.PID << ") at (" << face.pos.x << ", " << face.pos.y << ", " << face.pos.width << ", " << face.pos.height << ") ";
        }

        std::cout << std::endl;
        cv::imshow( "Tracker", mat );

        if( cv::waitKey( 40 ) >= 0 ) break;
        ++frameID;
    }

    return 0;

}
