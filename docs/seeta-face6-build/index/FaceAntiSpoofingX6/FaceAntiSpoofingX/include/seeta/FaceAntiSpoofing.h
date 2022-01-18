#ifndef _FACEANTISPOOFING_H_
#define _FACEANTISPOOFING_H_

#include "seeta/Common/Struct.h"

#include <string>
#include <vector>


namespace seeta
{
    namespace v6
    {

        class FaceAntiSpoofing {
        public:
            /*
             * 活体识别状态
             */
            enum Status
            {
                REAL = 0,   ///< 真实人脸
                SPOOF = 1,  ///< 攻击人脸（假人脸）
                FUZZY = 2,  ///< 无法判断（人脸成像质量不好）
                DETECTING = 3,  ///< 正在检测
            };

            enum Property {
                PROPERTY_NUMBER_THREADS = 4,
                PROPERTY_ARM_CPU_MODE = 5
            };


            /**
             * \brief 加载模型文件
             * \param setting 模型文件, 0-局部活体检测文件（必选），1-全局活体检测文件（可选）
             */
            SEETA_API explicit FaceAntiSpoofing( const seeta::ModelSetting &setting );
            SEETA_API ~FaceAntiSpoofing();


            /**
             * \brief 检测活体
             * \param [in] image 输入图像，需要 RGB 彩色通道
             * \param [in] face 要识别的人脸位置
             * \param [in] points 要识别的人脸特征点
             * \return 人脸状态 @see Status
             * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceAntiSpoofing 的对象分别调用检测函数
             * \note 当前版本可能返回 REAL, SPOOF, FUZZY
             * \see SeetaImageData, SeetaRect, PointF, Status
             */
            SEETA_API Status Predict( const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points ) const;

            /**
            * \brief 检测活体（Video模式）
            * \param [in] image 输入图像，需要 RGB 彩色通道
            * \param [in] face 要识别的人脸位置
            * \param [in] points 要识别的人脸特征点
            * \return 人脸状态 @see Status
            * \note 此函数不支持多线程调用，在多线程环境下需要建立对应的 FaceAntiSpoofing 的对象分别调用检测函数
            * \note 需要输入连续帧序列，当需要输入下一段视频是，需要调用 ResetVideo 重置检测状态
            * \note 当前版本可能返回 REAL, SPOOF, DETECTION
            * \see SeetaImageData, SeetaRect, PointF, Status
            */
            SEETA_API Status PredictVideo( const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points ) const;

            /**
             * \brief 重置 Video，开始下一次 PredictVideo 识别
             */
            SEETA_API void ResetVideo();

            /**
             * \brief 获取活体检测内部分数
             * \param [out] clarity 输出人脸质量分数
             * \param [out] reality 真实度
             * \note 获取的是上一次调用 Predict 或 PredictVideo 接口后内部的阈值
             */
            SEETA_API void GetPreFrameScore( float *clarity = nullptr, float *reality = nullptr );

            /**
             * 设置 Video 模式中，识别视频帧数，当输入帧数为该值以后才会有返回值
             * \param [in] number 视频帧数
             */
            SEETA_API void SetVideoFrameCount( int32_t number );

            /**
             * \return 获取视频帧数设置
             */
            SEETA_API int32_t GetVideoFrameCount() const;

            /**
             * 设置阈值
             * \param [in] clarity 清晰度阈值
             * \param [in] reality 活体阈值
             * \note clarity 越高要求输入的图像质量越高，reality 越高对识别要求越严格
             * \note 默认阈值为 0.3, 0.8
             */
            SEETA_API void SetThreshold( float clarity, float reality );
			
			 /**
             * 设置全局阈值
             * \param [in] box_thresh 全局检测阈值
             * \note 默认阈值为 0.8
             */
			 SEETA_API void SetBoxThresh(float box_thresh);
			 
			 SEETA_API float GetBoxThresh()const;

            /**
             * 获取阈值
             * \param [out] clarity 清晰度阈值
             * \param [out] reality 活体阈值
             */
            SEETA_API void GetThreshold( float *clarity = nullptr, float *reality = nullptr ) const;

            SEETA_API void set(Property property, double value);

            SEETA_API double get(Property property) const;
        private:
            FaceAntiSpoofing( const FaceAntiSpoofing & ) = delete;
            const FaceAntiSpoofing &operator=( const FaceAntiSpoofing & ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif
