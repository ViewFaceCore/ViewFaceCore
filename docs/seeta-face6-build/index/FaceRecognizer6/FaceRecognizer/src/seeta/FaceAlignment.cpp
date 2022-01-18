//
// Created by kier on 19-7-31.
//

#include <orz/utils/log.h>
#include "FaceAlignment.h"

#include "transform.h"

#include "api/cpp/intime.h"
#include <map>


namespace seeta {
    class FaceAlignment::Data {
    public:
        using MeanShape = std::vector<ts::Vec2D<float>>;
        using MeanShapeGroup = std::vector<MeanShape>;

        Data() {
            template_mean_shape_group = {
                    {{51.642f, 50.115f}, {57.617f, 49.990f}, {35.740f, 69.007f}, {51.157f, 89.050f}, {57.025f, 89.702f}},
                    {{45.031f, 50.118f}, {65.568f, 50.872f}, {39.677f, 68.111f}, {45.177f, 86.190f}, {64.246f, 86.758f}},
                    {{39.730f, 51.138f}, {72.270f, 51.138f}, {56.000f, 68.493f}, {42.463f, 87.010f}, {69.537f, 87.010f}},
                    {{46.845f, 50.872f}, {67.382f, 50.118f}, {72.737f, 68.111f}, {48.167f, 86.758f}, {67.236f, 86.190f}},
                    {{54.796f, 49.990f}, {60.771f, 50.115f}, {76.673f, 69.007f}, {55.388f, 89.702f}, {61.257f, 89.050f}},
            };
            template_size = 112;
        }

        MeanShapeGroup get_mean_shape_group(int size) {
            if (size == template_size) return template_mean_shape_group;
            auto it = ready_group.find(size);
            if (it != ready_group.end()) {
                return it->second;
            }
            MeanShapeGroup mean_shape_group = template_mean_shape_group;
            auto scale = float(size) / template_size;
            for (auto &mean_shape : mean_shape_group) {
                for (auto &point : mean_shape) {
                    point[0] *= scale;
                    point[1] *= scale;
                }
            }
            ready_group.insert(std::make_pair(size, mean_shape_group));
            return std::move(mean_shape_group);
        }

    private:
        std::map<int, MeanShapeGroup> ready_group;
        MeanShapeGroup template_mean_shape_group;
        int template_size;
    };

    FaceAlignment::FaceAlignment(const std::string &mode, int width, int height, int N) {
        if (mode == "single") {
            m_mode = SINGLE;
            if (N != 5) {
                ORZ_LOG(orz::ERROR) << "Single alignment only support 5 points" << orz::crash;
            }
            m_final_width = width;
            m_final_height = height;
            m_n = N;
        } else if (mode == "multi") {
            m_mode = MULTI;
            if (N != 5) {
                ORZ_LOG(orz::ERROR) << "Multi alignment only support 5 points" << orz::crash;
            }
            if (width != height) {
                ORZ_LOG(orz::ERROR) << "Multi alignment width and height must be same, got {"
                                    << width << ", " << height << "}" << orz::crash;
            }
            m_final_width = width;
            m_final_height = height;
            m_n = N;
        } else if (mode == "arcface") {
            m_mode = ARCFACE;
            if (N != 5) {
                ORZ_LOG(orz::ERROR) << "Arcface alignment only support 5 points" << orz::crash;
            }
            // if (width != height || width != 112) {
            //     ORZ_LOG(orz::ERROR) << "Arcface alignment width and height must be [112, 112], got ["
            //                         << width << ", " << height << "]" << orz::crash;
            // }
            m_final_width = width;
            m_final_height = height;
            m_n = N;
        } else {
            ORZ_LOG(orz::ERROR) << "Not supported alignment version: " << mode << orz::crash;
        }
        m_data = new Data;
        m_mode_string = mode;
    }

    FaceAlignment::~FaceAlignment() {
        delete m_data;
    }

    int FaceAlignment::crop_width() const {
        return m_final_width;
    }

    int FaceAlignment::crop_height() const {
        return m_final_height;
    }

    void FaceAlignment::crop_face(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const {
        if (m_mode == SINGLE) {
            crop_single(image, points, face);
        } else if (m_mode == MULTI) {
            crop_multi(image, points, face);
        } else if (m_mode == ARCFACE) {
            crop_arcface(image, points, face);
        } else {
            crop_single(image, points, face);
        }
    }

    void
    FaceAlignment::crop_single(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const {
        // parameters will be safe by caller
        std::vector<ts::Vec2D<float>> mean_shape = {
                {89.3095f,  72.9025f},
                {169.3095f, 72.9025f},
                {127.8949f, 127.0441f},
                {96.8796f,  184.8907f},
                {159.1065f, 184.7601f},
        };
        int width = 256;
        int height = 256;
        std::vector<ts::Vec2D<float>> landmarks = {
                {float(points[0].x), float(points[0].y)},
                {float(points[1].x), float(points[1].y)},
                {float(points[2].x), float(points[2].y)},
                {float(points[3].x), float(points[3].y)},
                {float(points[4].x), float(points[4].y)},
        };

        auto M = transform2d(landmarks, mean_shape);
        auto shift = ts::affine::translate<float>(-float(m_final_width - width) / 2, -float(m_final_height - height) / 2);
        ts::stack(shift, M);
        M = shift;

        auto tensor_image = ts::api::tensor::build(TS_UINT8, {image.height, image.width, image.channels}, image.data);
        auto tensor_affine = ts::api::tensor::build(TS_FLOAT32, {3, 3}, M.data());

        auto tensor_patch = ts::api::intime::affine_sample2d(tensor_image, {m_final_height, m_final_width},
                                                             tensor_affine, 0, 0, ts::api::intime::ResizeMethod::BILINEAR);
        tensor_patch.sync_cpu();

        memcpy(face.data, tensor_patch.data(), tensor_patch.count());
    }

    static float operator^(const ts::Vec2D<float> &lhs, const ts::Vec2D<float> &rhs) {
        auto dx = lhs[0] - rhs[0];
        auto dy = rhs[1] - rhs[1];
        return std::sqrt(dx * dx + dy * dy);
    }

    void FaceAlignment::crop_multi(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const {
        SimilarityTransform2D transform;
        std::vector<ts::Vec2D<float>> landmarks = {
                {float(points[0].x), float(points[0].y)},
                {float(points[1].x), float(points[1].y)},
                {float(points[2].x), float(points[2].y)},
                {float(points[3].x), float(points[3].y)},
                {float(points[4].x), float(points[4].y)},
        };
        float min_error = FLT_MAX;
        ts::Trans2D<float> min_M;

        auto src = m_data->get_mean_shape_group(m_final_width);

        for (size_t i = 0; i < src.size(); ++i) {
            auto &meanshape = src[i];
            transform.estimate(landmarks, meanshape);
            auto M = transform.params;
            std::vector<ts::Vec2D<float>> results(landmarks.size());
            for (size_t j = 0; j < results.size(); ++j) {
                results[j] = ts::transform(M, landmarks[j]);
            }
            float local_error = 0;
            for (size_t j = 0; j < results.size(); ++j) {
                local_error += results[j] ^ meanshape[j];
            }
            if (local_error < min_error) {
                min_error = local_error;
                min_M = M;
            }
        }
        auto M = min_M;
        M = ts::affine::inverse(M);

        auto tensor_image = ts::api::tensor::build(TS_UINT8, {image.height, image.width, image.channels}, image.data);
        auto tensor_affine = ts::api::tensor::build(TS_FLOAT32, {3, 3}, M.data());

        auto tensor_patch = ts::api::intime::affine_sample2d(tensor_image, {m_final_height, m_final_width},
                                                             tensor_affine, 0, 0, ts::api::intime::ResizeMethod::BILINEAR);
        tensor_patch.sync_cpu();

        memcpy(face.data, tensor_patch.data(), tensor_patch.count());
    }
    void
    FaceAlignment::crop_arcface(const SeetaImageData &image, const SeetaPointF *points, SeetaImageData &face) const {
        // parameters will be safe by caller
        std::vector<ts::Vec2D<float>> mean_shape = {
                {38.2946f, 51.6963f},
                {73.5318f, 51.5014f},
                {56.0252f, 71.7366f},
                {41.5493f, 92.3655f},
                {70.7299f, 92.2041f},
        };

        std::vector<ts::Vec2D<float>> landmarks = {
                {float(points[0].x), float(points[0].y)},
                {float(points[1].x), float(points[1].y)},
                {float(points[2].x), float(points[2].y)},
                {float(points[3].x), float(points[3].y)},
                {float(points[4].x), float(points[4].y)},
        };

        auto M = transform2d(landmarks, mean_shape);

        auto tensor_image = ts::api::tensor::build(TS_UINT8, {image.height, image.width, image.channels}, image.data);
        auto tensor_affine = ts::api::tensor::build(TS_FLOAT32, {3, 3}, M.data());

        auto tensor_patch = ts::api::intime::affine_sample2d(tensor_image, {m_final_height, m_final_width},
                                                             tensor_affine, 0, 0, ts::api::intime::ResizeMethod::BILINEAR);
        tensor_patch.sync_cpu();

        memcpy(face.data, tensor_patch.data(), tensor_patch.count());
    }
}