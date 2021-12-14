//
// Created by kier on 19-7-24.
//

#ifndef SEETA_QULITY_EVALUATOR_QULITY_EVALUATOR_H
#define SEETA_QULITY_EVALUATOR_QULITY_EVALUATOR_H

#define SEETA_FACE_QULITY_EVALUATOR_MAJOR_VERSION 3
#define SEETA_FACE_QULITY_EVALUATOR_MINOR_VERSION 0
#define SEETA_FACE_QULITY_EVALUATOR_SINOR_VERSION 0

#include "Struct.h"
#include "QualityStructure.h"

namespace seeta {
    namespace v3 {
        enum QualityAttribute : int32_t {
            BRIGHTNESS = 0x00,
	    CLARITY = 0x01,
	    INTEGRITY = 0x02,
	    POSE = 0x03,
            RESOLUTION = 0x04,
            POSE_EX = 0x05, 
        };
        class QualityAssessor {
        public:
            SEETA_API explicit QualityAssessor();
            SEETA_API ~QualityAssessor();

            /**
             * Add rule to quality assessor
             * @param attr attr id
             * @param must_high if treat MEDIUM as low
             * @note raise std::exception if attr not recognized
             */
            SEETA_API void add_rule(int32_t attr, bool must_high=false);

            /**
             * Add rule to quality assessor
             * @param attr attr id
             * @param model model setting if quality assessor need model
             * @param must_high if treat MEDIUM as low
             * @note raise std::exception if attr not recognized
             */
            SEETA_API void add_rule(int32_t attr, const SeetaModelSetting &model, bool must_high=false);

            /**
             * Add rule to quality assessor
             * @param attr attr id
             * @param rule rule
             * @param must_high if treat MEDIUM as low
             * @note this object just borrow object, please keep rule effective
             */
            SEETA_API void add_rule(int32_t attr, QualityRule *rule, bool must_high=false);

            /**
             * Remove added attr, happen nothing if not added
             * @param attr
             */
            SEETA_API void remove_rule(int32_t attr);

            /**
             *
             * @param attr
             * @return if has rule of attr
             */
            SEETA_API bool has_rule(int32_t attr);

            /**
             * How many MEDIUM can be accepted in all attributes, default is 0
             * @param limit
             */
            SEETA_API void set_medium_limit(int32_t limit);

            /**
             * Assess image with face
             * @param image original image
             * @param face face location
             * @param points landmark on face
             * @param N how many landmark on face given, must be 5 in this version
             * @return true if quality OK
             * @note evaluate will short return, if got any LOW quality attribute
             *       if you want evaluate each attr, call @see feed instead
             */
            SEETA_API bool evaluate(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    int32_t N
                    );

            /**
             * Assess image with face
             * @param image original image
             * @param face face location
             * @param points landmark on face
             * @param N how many landmark on face given, must be 5 in this version
             * @param result return error attribute info 
             * @return true if quality OK
             * @note evaluate will short return, if got any LOW quality attribute
             *       if you want evaluate each attr, call @see feed instead
             */
            SEETA_API bool evaluate(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF *points,
                    int32_t N,
                    QualityResultExArray &result
                    );

            /**
             * Call each quality rule, then call @see query to get each attr quality
             * @param image original image
             * @param face face location
             * @param points landmark on face
             * @param N how many landmark on face given, must be 5 in this version
             * @note evaluate will short return, if got any LOW quality attribute
             *       if you want evaluate each attr, call @see feed instead
             */
            SEETA_API void feed(
                    const SeetaImageData &image,
                    const SeetaRect &face,
                    const SeetaPointF* points,
                    int32_t N);

            /**
             * Query attr result, after evaluate or feed
             * @param attr
             * @return quality result
             * @note if attr not added, return {LOW, 0}
             */
            SEETA_API QualityResult query(int32_t attr) const;

            /**
             * Disable an added attr
             * @param attr @sa QualityAttribute
             */
            SEETA_API void disable(int32_t attr);

            /**
             * Enable attr which disabled by @see disable
             * @param attr @sa QualityAttribute
             */
            SEETA_API void enable(int32_t attr);

        private:
            QualityAssessor(const QualityAssessor &) = delete;
            const QualityAssessor &operator=(const QualityAssessor&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v3;
}

#endif //SEETA_QULITY_EVALUATOR_QULITY_EVALUATOR_H
