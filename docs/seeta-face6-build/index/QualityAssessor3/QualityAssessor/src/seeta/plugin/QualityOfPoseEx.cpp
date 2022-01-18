//
// Created by kier on 2019-07-24.
//

#include "seeta/QualityOfPoseEx.h"
#include <seeta/PoseEstimator.h>

#include <cmath>
#include <cfloat>
#include <climits>
#include <assert.h>
 
namespace seeta {
    
    QualityOfPoseEx::QualityOfPoseEx(const SeetaModelSetting &setting) {
        m_data = new PoseEstimator(setting);
        m_yaw_low = 25;
        m_pitch_low = 20;
        m_roll_low = 33.33;

        m_yaw_high = 10;
        m_pitch_high = 10;
        m_roll_high = 16.67;
    }

    QualityOfPoseEx::~QualityOfPoseEx() {
        delete reinterpret_cast<PoseEstimator *>(m_data);
    }


    static int quality_level(double score, float thresh0, float thresh1) {
        if (score < thresh0) {
            return 2;
        } else if (score < thresh1) {
            return 1;
        } else {
            return 0;
        }
    }

    QualityResult QualityOfPoseEx::check(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points,
                                       const int32_t N) {
        assert(points != nullptr && N == 5);
        PoseEstimator *data = reinterpret_cast<PoseEstimator *>(m_data);

        float yaw, pitch, roll;
        yaw = pitch = roll = 0.0;
 
        data->Estimate(image, face, &yaw, &pitch, &roll);

        yaw = fabs(yaw);
        pitch = fabs(pitch);
        roll = fabs(roll);

        float yaw_score = (90 - yaw - 30) / 60;
        float pitch_score = (90 - pitch - 45) / 45;
        if(yaw_score < 0)
        {
            yaw_score = 0;
        }

        if(pitch_score < 0)
        {
            pitch_score = 0;
        }

        float pose_score = 0.5f * yaw_score + 0.5f * pitch_score;

        //int quality_roll = quality_level(roll, m_roll_high, m_roll_low);
        int quality_yaw = quality_level(yaw, m_yaw_high, m_yaw_low);
        int quality_pitch = quality_level(pitch, m_pitch_high, m_pitch_low);

        //int quality = std::min(std::min(quality_roll, quality_yaw), quality_pitch);
        int quality = std::min(quality_yaw, quality_pitch);

        return QualityResult(QualityLevel(quality), pose_score);

    }

	bool QualityOfPoseEx::check(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points,
                                       const int32_t N, float& _yaw, float & _pitch, float &_roll) {
        assert(points != nullptr && N == 5);
        PoseEstimator *data = reinterpret_cast<PoseEstimator *>(m_data);

        float yaw, pitch, roll;
        yaw = pitch = roll = 0.0;
 
        data->Estimate(image, face, &yaw, &pitch, &roll);
		
		_yaw = yaw;
		_pitch = pitch;
		_roll = roll;
		
		return true;
    }
	
    float QualityOfPoseEx::get(PROPERTY property)
    {
        switch(property)
        {
            case YAW_LOW_THRESHOLD:
                return m_yaw_low;
            case PITCH_LOW_THRESHOLD:
                return m_pitch_low;
            case ROLL_LOW_THRESHOLD:
                return m_roll_low;
            case YAW_HIGH_THRESHOLD:
                return m_yaw_high;
            case PITCH_HIGH_THRESHOLD:
                return m_pitch_high;
            case ROLL_HIGH_THRESHOLD:
                return m_roll_high;
            default:
                return 0.0;

        }
        return 0.0;
    }

    void QualityOfPoseEx::set(PROPERTY property, float value)
    {
        switch(property)
        {
            case YAW_LOW_THRESHOLD:
                m_yaw_low = value;
                break;
            case PITCH_LOW_THRESHOLD:
                m_pitch_low = value;
                break;
            case ROLL_LOW_THRESHOLD:
                m_roll_low = value;
                break;
            case YAW_HIGH_THRESHOLD:
                m_yaw_high = value;
                break;
            case PITCH_HIGH_THRESHOLD:
                m_pitch_high = value;
                break;
            case ROLL_HIGH_THRESHOLD:
                m_roll_high = value;
                break;
            default:
                break;

        }
        return;
    }

}
