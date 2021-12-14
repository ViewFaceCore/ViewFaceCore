#ifndef SEETA_GENDER_PREDICTOR_H
#define SEETA_GENDER_PREDICTOR_H

#include "Common/Struct.h"

#define SEETA_AGE_PREDICTOR_MAJOR_VERSION 6
#define SEETA_AGE_PREDICTOR_MINOR_VERSION 0
#define SEETA_AGE_PREDICTOR_SINOR_VERSION 0

namespace seeta {
    namespace v6 {
        class GenderPredictor {
        public:

			enum Property {
                 PROPERTY_NUMBER_THREADS = 4,
                 PROPERTY_ARM_CPU_MODE = 5
            };

			enum GENDER
			{
				MALE,
				FEMALE
			};

            SEETA_API explicit GenderPredictor(const SeetaModelSetting &setting);
            SEETA_API ~GenderPredictor();

            SEETA_API  int GetCropFaceWidth() const;
            SEETA_API  int GetCropFaceHeight() const;
            SEETA_API  int GetCropFaceChannels() const;
            SEETA_API  bool CropFace(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

            SEETA_API bool PredictGender(const SeetaImageData &image, GENDER &gender) const;

			SEETA_API bool PredictGenderWithCrop(const SeetaImageData &image, const SeetaPointF *points, GENDER &gender) const;

            SEETA_API void set(Property property, double value);

            SEETA_API double get(Property property) const;

        private:
            GenderPredictor(const GenderPredictor &) = delete;
            const GenderPredictor &operator=(const GenderPredictor&) = delete;

        private:
            class Implement;
            Implement *m_impl;
        };
    }
    using namespace v6;
}

#endif //SEETA_GENDER_PREDICTOR_H
