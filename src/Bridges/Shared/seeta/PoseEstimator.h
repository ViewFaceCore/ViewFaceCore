#pragma once

#include "Common/Struct.h"
#include <string>
#include <vector>

#define SEETA_POSE_ESTIMATOR_MAJOR_VERSION 6
#define SEETA_POSE_ESTIMATOR_MINOR_VERSION 0
#define SEETA_POSE_ESTIMATOR_SINOR_VERSION 0

namespace seeta
{
    namespace v6
    {
        class PoseEstimator {
        public:
            enum Axis
            {
                YAW     = 0,
                PITCH   = 1,
                ROLL    = 2,
            };

            enum Property {
                PROPERTY_NUMBER_THREADS = 4,
                PROPERTY_ARM_CPU_MODE = 5
            }; 

            /**
             * \brief initialize `PoseEstimator`
             * \param setting one specifc model, or zero model
             */
            SEETA_API explicit PoseEstimator( const seeta::ModelSetting &setting = seeta::ModelSetting() );

            SEETA_API ~PoseEstimator();


            /**
             * \brief Feed image to `PoseEstimator`
             * \param image The orginal image
             * \param face The face location
             */
            SEETA_API void Feed( const SeetaImageData &image, const SeetaRect &face ) const;

            /**
             * \brief get angle on given axis
             * \param axis \sa `Axis`: YAW, PITCH, or ROLL
             * \return angle on given axis
             * \note Must `Feed` image and face first.
             */
            SEETA_API float Get( Axis axis ) const;

            /**
             * \brief Get angle from given face on image
             * \param image The orginal image
             * \param face The face location
             * \param [out] yaw angle on axis yaw
             * \param [out] pitch angle on axis pitch
             * \param [out] roll angle on axis roll
             * \note yaw, pitch or roll can be nullptr
             */
            SEETA_API void Estimate( const SeetaImageData &image, const SeetaRect &face, float *yaw, float *pitch, float *roll ) const;


            SEETA_API void set(Property property, double value);

            SEETA_API double get(Property property) const;


        private:
            PoseEstimator( const PoseEstimator &other ) = delete;
            const PoseEstimator &operator=( const PoseEstimator &other ) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

