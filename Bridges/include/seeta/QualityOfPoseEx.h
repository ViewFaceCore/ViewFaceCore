//
// Created by kier on 2019-07-24.
//

#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFPOSEEX_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFPOSEEX_H

#include "QualityStructure.h"

namespace seeta {
    namespace v3 {

        class QualityOfPoseEx : public QualityRule {
        public:

            enum PROPERTY
            {
                YAW_LOW_THRESHOLD = 0,
                YAW_HIGH_THRESHOLD = 1,
                PITCH_LOW_THRESHOLD = 2,
                PITCH_HIGH_THRESHOLD = 3,
                ROLL_LOW_THRESHOLD = 4, 
                ROLL_HIGH_THRESHOLD = 5
            };

            using self = QualityOfPoseEx;
            using supper = QualityRule;

            /**
             * Construct with recommend parameters
             */
            SEETA_API QualityOfPoseEx(const SeetaModelSetting &setting);

            SEETA_API ~QualityOfPoseEx() override;

            SEETA_API QualityResult check(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    const int32_t N) override;
					
            SEETA_API float get(PROPERTY property);
 
            SEETA_API void set(PROPERTY property, float value);
        private:
            QualityOfPoseEx(const QualityOfPoseEx &) = delete;
            QualityOfPoseEx &operator=(const QualityOfPoseEx &) = delete;

        private:
            void *m_data;
            float m_yaw_low;
            float m_pitch_low;
            float m_roll_low;

            float m_yaw_high;
            float m_pitch_high;
            float m_roll_high;
        };
    }
    using namespace v3;
}

#endif //SEETA_QUALITYEVALUATOR_QUALITYOFPOSEEX_H
