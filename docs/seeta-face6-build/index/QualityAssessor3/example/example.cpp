//
// Created by kier on 19-4-24.
//
#include "seeta/FaceDetector.h"
//#include "seeta/PointDetector.h"

#include "seeta/FaceLandmarker.h"
#include "seeta/QualityAssessor.h"

#include "seeta/QualityOfPoseEx.h"
#include "seeta/QualityOfPose.h"
#include "seeta/QualityOfBrightness.h"
#include "seeta/QualityOfResolution.h"
#include "seeta/QualityOfClarity.h"
#include "seeta/QualityOfIntegrity.h"

#include "seeta/QualityOfLBN.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>


#include "CVStatisticsWindow.h"

cv::Scalar red(255, 0, 0);
cv::Scalar blue(0, 0, 255);

void draw_rectangle(cv::Mat &image, const SeetaRect &rect, const cv::Scalar &color)
{
	cv::rectangle(image, cv::Rect(rect.x, rect.y, rect.width, rect.height), color, 2);
}

void put_text(cv::Mat &image, cv::Point & point, std::string message)
{
	cv::putText(image, message, point, 1, 2.0, blue);
}


static unsigned int hextochar(unsigned char c)
{
     if((c >= 'A') && (c <= 'F'))
    {
        return 10 + c - 'A';
    }else if((c >= 'a') && (c <= 'f'))
    {
        return 10 + c - 'a';
    }else if((c >= '0') && (c <= '9'))
    {
        return c - '0';
    }
    return 0;
}

static unsigned char getcolor(const std::string &str)
{
    unsigned char ret = 0;
    if(str.length() != 2)
        return ret;
    ret = hextochar(str[0]) * 16 + hextochar(str[1]);
    return ret;
}

static cv::Scalar getrgb(const std::string &str)
{
    if(str.length() != 6)
    {
        return cv::Scalar(0,0,0);
    }

    std::string strr = str.substr(0,2);
    std::string strg = str.substr(2,2);
    std::string strb = str.substr(4,2);
 
    int r = getcolor(strr); 
    int g = getcolor(strg); 
    int b = getcolor(strb); 
    return cv::Scalar(r,g,b);
}

//example 006666
int main(int argc, char ** argv) {

        std::string strcolor;

        
        if(argc == 2)
        {
            strcolor = argv[1];
        }
        

        /*
        if(argc != 2)
        {
            std::cout << "parameter error!" << std::endl;
            std::cout << "usage ./example imagefilename" << std::endl;
            return -1;
        }
        */

        //std::string imagefile(argv[1]);
	seeta::FaceDetector FD(seeta::ModelSetting("./model/faceboxes_2019-4-29_0.90.sta"));
	//seeta::PointDetector PD(seeta::ModelSetting("./model/SeetaPointDetector5.0.pts5.sta"));

	seeta::FaceLandmarker PD(seeta::ModelSetting("./model/SeetaFaceLandmarker5.0.tsm.sta"));
	
        seeta::ModelSetting posemodel;
        posemodel.set_device(SEETA_DEVICE_CPU);
        posemodel.set_id(0); 
        posemodel.append("./model/PoseEstimation1.1.0.json");
        //landmark.append("./model/FL_mask_cnn6mb1_pts5.json");


        seeta::QualityOfPoseEx *poseex = new seeta::QualityOfPoseEx(posemodel);
        poseex->set(seeta::QualityOfPoseEx::YAW_LOW_THRESHOLD, 20);
        poseex->set(seeta::QualityOfPoseEx::YAW_HIGH_THRESHOLD, 10);

        poseex->set(seeta::QualityOfPoseEx::PITCH_LOW_THRESHOLD, 20);
        poseex->set(seeta::QualityOfPoseEx::PITCH_HIGH_THRESHOLD, 10);


	seeta::ModelSetting lbnmodel;
        lbnmodel.set_device(SEETA_DEVICE_CPU);
        lbnmodel.set_id(0); 

        //lbnmodel.append("./model/_acc858873_squeezenetV19_227_multyLabel_1107.json");
        lbnmodel.append("./model/_loss017016085_squeezenetV21_blur1_2thDB_370000_1115.json");
        seeta::QualityOfLBN lbn(lbnmodel);
        lbn.set(seeta::QualityOfLBN::PROPERTY_BLUR_THRESH, 0.80);
        seeta::ModelSetting setting3;
        setting3.set_id(0); 
        setting3.set_device( SEETA_DEVICE_CPU );

        setting3.append( "./model/SeetaFaceLandmarkerV5.Pts68.sta" );
        seeta::FaceLandmarker PD68( setting3 );

        //setting3.append( "./model/VIPLPointDetector5.0.pts68.json" );
        //seeta::PointDetector PD68( setting3 );

        seeta::ModelSetting setting4;
        setting4.set_device(SEETA_DEVICE_CPU);
        setting4.set_id(0); 
        setting4.append("./model/FL_mask_cnn6mb1_pts5.json");
        seeta::FaceLandmarker FL(setting4);


	seeta::QualityAssessor qa;
	qa.add_rule(seeta::INTEGRITY);
	qa.add_rule(seeta::RESOLUTION);
	qa.add_rule(seeta::BRIGHTNESS);
	qa.add_rule(seeta::CLARITY);
	qa.add_rule(seeta::POSE);
	//qa.set_medium_limit(1);

	qa.add_rule(seeta::POSE_EX, poseex, true);


        //////////////////////////
        CVStatisticsWindow evstat;

        if(strcolor.length() == 6)
        {
            evstat.set_color(getrgb(strcolor));
        }
        evstat.add_name("POSE", -1);
        evstat.add_name("CLARITY", -1);
        evstat.add_name("INTEGRITY", -1);
        evstat.add_name("RESOLUTION", -1);
        evstat.add_name("BRIGHTNESS", -1);

        evstat.add_name("POSE_EX", -1);
        //////////////////////////

        int landmark_num = FL.number();
        std::vector<SeetaPointF> landmarks;
        landmarks.resize(landmark_num);
        std::vector<int> masks;
        masks.resize(landmark_num);

        //static cv::Scalar blue(255, 0, 0);
        static cv::Scalar green(0, 255, 0);
        static cv::Scalar red(0, 0, 255);

 
	cv::VideoCapture capture(0);
	cv::Mat mat;
	while (capture.isOpened())
        //do
	{
		capture >> mat;
		if (!mat.isContinuous()) break;

                //mat = cv::imread("hiha/2.bmp"); 
                //mat = cv::imread(imagefile.c_str()); 
		SeetaImageData image;
		image.width = mat.cols;
		image.height = mat.rows;
		image.channels = mat.channels();
		image.data = mat.data;

		auto face_array = FD.detect(image);
            
                if(face_array.size == 0)
                {
                        evstat.set_value("BRIGHTNESS", -1);
                        evstat.set_value("RESOLUTION", -1);
                        evstat.set_value("CLARITY", -1);
                        evstat.set_value("INTEGRITY", -1);
                        evstat.set_value("POSE", -1);
                        evstat.set_value("POSE_EX", -1);

                        evstat.set_light_value(-1);
                        evstat.set_blur_value(-1);
                        evstat.set_noise_value(-1);
                        std::vector<int> tmpmask;
                        tmpmask.resize(5, -1);
                        evstat.set_mask_values(tmpmask);
                         
                }

		for (int i = 0; i < face_array.size; ++i)
		{
                        if(i>0)
                        {
                            break;
                        }
			SeetaRect& face = face_array.data[i].pos;
			SeetaPointF points[5];


			//PD.Detect(image, face, points);
			PD.mark(image, face, points);

			qa.feed(image, face, points, 5);
			auto result1 = qa.query(seeta::BRIGHTNESS);
			auto result2 = qa.query(seeta::RESOLUTION);
			auto result3 = qa.query(seeta::CLARITY);
			auto result4 = qa.query(seeta::INTEGRITY);
			auto result5 = qa.query(seeta::POSE);

			auto result = qa.query(seeta::POSE_EX);

                        
                        evstat.set_value("BRIGHTNESS", (int)result1.level); 
                        evstat.set_value("RESOLUTION", (int)result2.level); 
                        evstat.set_value("CLARITY", (int)result3.level); 
                        evstat.set_value("INTEGRITY", (int)result4.level); 
                        evstat.set_value("POSE", (int)result5.level); 
                        evstat.set_value("POSE_EX", (int)result.level); 


                        
			SeetaPointF points2[68];
                        memset( points2, 0, sizeof( SeetaPointF ) * 68 );
                        //PD68.Detect(image, face,points2);

                        PD68.mark(image, face,points2);
                        int light, blur, noise;
                        light = blur = noise = -1;

                        lbn.Detect( image, points2, &light, &blur, &noise );
                        evstat.set_light_value(light);
                        evstat.set_blur_value(blur);
                        evstat.set_noise_value(noise);
                        //std::cout << "light:" << light << ", blur:" << blur << ", noise:" << noise << std::endl;

                        FL.mark(image, face, landmarks.data(), masks.data());
                        evstat.set_mask_values(masks);

			draw_rectangle(mat, face , red);
                        for(int m=0; m<masks.size(); m++)
                        {
                             auto &point = landmarks[m];
                             cv::circle(mat, cv::Point(point.x, point.y), 3, (masks[m] ? red : green), -1);
                        }
		}

                evstat.update();
		cv::imshow("test", mat);
                evstat.imshow("stat"); 
		if (cv::waitKey(1) >= 27) break;
	}
        //while(0);

        return 0;
}
