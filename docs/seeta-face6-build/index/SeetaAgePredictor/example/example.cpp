#include "seeta/AgePredictor.h"
#include "seeta/Common/Struct.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include <fstream>
#include <chrono>

#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>

int main_video()
{
	int device_id = 0;
    std::string ModelPath = "./model/";
	seeta::ModelSetting FD_model;
	FD_model.append(ModelPath + "SeetaFaceDetector6.0.IPC.sta");
	FD_model.set_device(seeta::ModelSetting::CPU);
	FD_model.set_id(device_id);

	seeta::ModelSetting PD_model;
	PD_model.append(ModelPath + "SeetaFaceLandmarker5.0.pts5.tsm.sta");
	PD_model.set_device(seeta::ModelSetting::CPU);
	PD_model.set_id(device_id);

	
	seeta::FaceDetector FD(FD_model); //人脸检测的初始化

	seeta::FaceLandmarker FL(PD_model); //关键点检测模型初始化

	const char* model_path = "./model/SeetaAgePredictor2.0.CJF.ext.json";
	seeta::ModelSetting setting(model_path);
	seeta::AgePredictor AP(setting);
	
	std::string title = "AgePredictor";
	cv::namedWindow(title, cv::WINDOW_NORMAL);

	cv::VideoCapture vc(0);
	cv::Mat frame;
	cv::Mat canvas;


	while (vc.isOpened())
	{
			if (cv::waitKey(33) >= 0) break;

			vc >> frame;
			if (!frame.data) continue;
			cv::flip(frame, canvas, 1);

			SeetaImageData simg;
			simg.height = frame.rows;
			simg.width = frame.cols;
			simg.channels = frame.channels();
			simg.data = frame.data;

			auto infos = FD.detect(simg);

			int line_width = 4;
			
			std::ostringstream oss;
			for (int i=0; i<infos.size; i++)
			{
				float scale = infos.data[i].pos.width / 300.0;
				float line = 30;
				cv::rectangle(canvas, cv::Rect(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y, 
							  infos.data[i].pos.width, infos.data[i].pos.height), cv::Scalar(128, 0, 0), scale * line_width);
				
				SeetaPointF points[5];
				FL.mark(simg, infos.data[i].pos, points);
				
				int age;
				AP.PredictAgeWithCrop(simg, points, age);
				
				oss.str("");
				oss << "age:  " << age;
				cv::putText(canvas, oss.str(), cv::Point(canvas.cols - infos.data[i].pos.x - infos.data[i].pos.width, infos.data[i].pos.y - 10 * scale), 0, scale, cv::Scalar(0, 128, 0), scale * line_width);
			}

			cv::imshow(title, canvas);
	}

        return 0;


}

int main()
{
    return main_video();
}
