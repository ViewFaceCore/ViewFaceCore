#ifndef INC_SEETA_FACETRACKING_H
#define INC_SEETA_FACETRACKING_H

#include "Common/Struct.h"
#include "CTrackingFaceInfo.h"
#include <string>
#include <vector>


namespace seeta
{
    namespace v6
    {
        class FaceTracker {
        public:
            /**
             * \brief initialize FaceTracker with face detector model
             * \param setting model used by FaceDetector
             * \param video_width input video frame width
             * \param video_height input video frame height
             */
            SEETA_API explicit FaceTracker( const seeta::ModelSetting &setting, int video_width, int video_height );
            SEETA_API ~FaceTracker();

            SEETA_API void SetSingleCalculationThreads( int num );
            
            /**
             * Set tracking frame interval
             */
            SEETA_API void SetInterval(int interval); 

            /**
             * \brief 检测人脸
             * \param [in] image 输入图像，需要 RGB 彩色通道
             * \return 检测到的人脸（SeetaTrackingFaceInfo）数组
             * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceTracker 的对象分别调用检测函数
             * \see SeetaTrackingFaceInfo, SeetaImageData
             */
            SEETA_API SeetaTrackingFaceInfoArray Track( const SeetaImageData &image ) const;

            /**
             * \brief 检测人脸
             * \param [in] image 输入图像，需要 RGB 彩色通道
             * \param [in] frame_no 输入帧号，跟输出帧号有关
             * \return 检测到的人脸（SeetaTrackingFaceInfo）数组
             * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceTracker 的对象分别调用检测函数
             * \see SeetaTrackingFaceInfo, SeetaImageData
             * \note frame_no 小于0 则自动化 frame_no
             */
            SEETA_API SeetaTrackingFaceInfoArray Track( const SeetaImageData &image, int frame_no ) const;

            /**
             * \brief 设置最小人脸
             * \param [in] size 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
             * \note 最下人脸为 20，小于 20 的值会被忽略
             */
            SEETA_API void SetMinFaceSize( int32_t size );

            /**
             * \brief 获取最小人脸
             * \return 最小可检测的人脸大小，为人脸宽和高乘积的二次根值
             */
            SEETA_API int32_t GetMinFaceSize() const;


            SEETA_API void SetThreshold( float thresh );

            SEETA_API float GetThreshold() const;


	    SEETA_API void SetVideoStable(bool stable = true);

	    SEETA_API bool GetVideoStable() const;

            SEETA_API void SetVideoSize(int vidwidth, int vidheight);

            SEETA_API void Reset();           
        private:
            FaceTracker( const FaceTracker &other ) = delete;
            const FaceTracker &operator=( const FaceTracker &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif
