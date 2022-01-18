//
// Created by kier on 19-7-31.
//

#ifndef SEETA_FACERECOGNIZER_FACEALIGNMENT_H
#define SEETA_FACERECOGNIZER_FACEALIGNMENT_H

#include <string>
#include <memory>
#include "seeta/FaceRecognizer.h"

namespace seeta {
    class FaceAlignment {
    public:
        enum Mode {
            SINGLE = 0,
            MULTI = 1,
            ARCFACE = 2,
        };

        using self = FaceAlignment;
        using shared = std::shared_ptr<self>;

        /**
         *
         * @param mode in {single, multi}
         * @param width output width
         * @param height output height
         * 1. mode single use force 256x256 meanshape, {width, height} change the final shape, not meanshape
         * 2. mode multi use dynamic meanshape, {width, height} change the meanshape
         */
        FaceAlignment(const std::string &mode, int width, int height, int N);
        ~FaceAlignment();

        int crop_width() const;

        int crop_height() const;

        void crop_face(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

        shared clone() const {
            return std::make_shared<FaceAlignment>(m_mode_string, m_final_width, m_final_height, m_n);
        }

    private:
        void crop_single(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

        void crop_multi(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

        void crop_arcface(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const;

    private:
        std::string m_mode_string;
        Mode m_mode;
        int m_final_width;
        int m_final_height;
        int m_n;

        class Data;
        Data *m_data;

        FaceAlignment(const FaceAlignment &) = delete;
        FaceAlignment &operator=(const FaceAlignment &) = delete;

    };
}

#endif //SEETA_FACERECOGNIZER_FACEALIGNMENT_H
