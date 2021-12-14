#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFLBN_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFLBN_H


#include "Struct.h"
#include <string>
#include <vector>

#define SEETA_POSE_ESTIMATOR_MAJOR_VERSION 6
#define SEETA_POSE_ESTIMATOR_MINOR_VERSION 0
#define SEETA_POSE_ESTIMATOR_SINOR_VERSION 0

namespace seeta
{
    namespace v6
    {
        class QualityOfLBN {
        public:
            enum LIGHTSTATE 
            {
                BRIGHT    = 0,
                DARK      = 1
                //NORMAL    = 2
            };

            enum BLURSTATE 
            {
                CLEAR          = 0,
                BLUR       = 1
                //SERIOUSBLUR    = 2
            };

            enum NOISESTATE 
            {
                HAVENOISE    = 0,
                NONOISE      = 1
            };

            enum Property {
                PROPERTY_NUMBER_THREADS = 4,
                PROPERTY_ARM_CPU_MODE = 5,
                PROPERTY_LIGHT_THRESH = 10,
                PROPERTY_BLUR_THRESH = 11,
                PROPERTY_NOISE_THRESH = 12
            }; 

            /**
             * \brief initialize `QualityOfLBN`
             * \param setting one specifc model, or zero model
             */
            SEETA_API explicit QualityOfLBN( const seeta::ModelSetting &setting = seeta::ModelSetting() );

            SEETA_API ~QualityOfLBN();


            /**
             * \brief Get angle from given face on image
             * \param image The orginal image
             * \param points face location array,size must is 68
             * \param [out] LIGHTSTATE 
             * \param [out] BLURSTATE
             * \param [out] NOISESTATE
             * \note light, blur or noise can be nullptr
             */
            SEETA_API void Detect( const SeetaImageData &image, const SeetaPointF *points, int *light, int *blur, int *noise ) const;


            SEETA_API void set(Property property, double value);
            SEETA_API double get(Property property) const;


        private:
            QualityOfLBN( const QualityOfLBN &other ) = delete;
            const QualityOfLBN &operator=( const QualityOfLBN &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif
