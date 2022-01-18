//
// Created by kier on 19-8-30.
//

#ifndef INC_SEETA_C_FACEDETECTOR_H
#define INC_SEETA_C_FACEDETECTOR_H

#include "Common/CStruct.h"
#include "CFaceInfo.h"

struct SeetaFaceDetector;

enum seeta_v6_FaceDetector_Property {
    SEETA_PROPERTY_MIN_FACE_SIZE = 0,
    SEETA_PROPERTY_THRESHOLD = 1,
    SEETA_PROPERTY_MAX_IMAGE_WIDTH = 2,
    SEETA_PROPERTY_MAX_IMAGE_HEIGHT = 3,
    SEETA_PROPERTY_NUMBER_THREADS = 4,

    /*
     * -1 for default, 0 for big core, 1 for little core, 2 for all cores
     * @note **ONLY** work in ARM arch.
     */
    SEETA_PROPERTY_ARM_CPU_MODE = 0x101,
};

SEETA_C_API const char *seeta_v6_FaceDetector_error();

SEETA_C_API SeetaFaceDetector *seeta_v6_FaceDetector_new(const SeetaModelSetting *setting);

SEETA_C_API void seeta_v6_FaceDetector_delete(const SeetaFaceDetector *object);

SEETA_C_API SeetaFaceDetector *seeta_v6_FaceDetector_clone(const SeetaFaceDetector *object);

SEETA_C_API SeetaFaceInfoArray seeta_v6_FaceDetector_detect(const SeetaFaceDetector *object,
                                                const SeetaImageData *image);

SEETA_C_API void seeta_v6_FaceDetector_set(SeetaFaceDetector *object,
                               seeta_v6_FaceDetector_Property property,
                               double value);

SEETA_C_API double seeta_v6_FaceDetector_get(const SeetaFaceDetector *object,
                                 seeta_v6_FaceDetector_Property property);

#endif //INC_SEETA_C_FACEDETECTOR_H
