//
// Created by kier on 2019-07-24.
//

#include "seeta/QualityOfPose.h"

#include <cmath>
#include <cfloat>
#include <climits>
#include <assert.h>

#define M_PI 3.14159265358979323846
namespace seeta {
    namespace {
        struct Data {
            // 定义眼中心和嘴中心连线，为眼口线
            // using two eyes calculate angle
            // 表示人头偏离垂直向上方向的角度，通过与双眼角度垂直获取
            float roll0 = 1 / 6.;
            float roll1 = 1 / 3.;

            // using distance of nose top to face line
            // 表示鼻尖点，到眼口线的距离，单位：倍眼距
            float yaw0 = 0.2;
            float yaw1 = 0.5;

            // use scale of projection point on face line to center of eyes
            // 表示鼻尖点投影在眼口线上的点，偏离正常投影点的距离，单位：倍眼口线长度
            float pitch0 = 0.2;
            float pitch1 = 0.5;

            // 鼻尖点正常投影在眼口线上的点到眼中心的距离，单位：倍眼口线长度
            float nose_center = 0.5;
        };
    }
    QualityOfPose::QualityOfPose() {
        m_data = new Data;
    }

    QualityOfPose::~QualityOfPose() {
        delete reinterpret_cast<Data *>(m_data);
    }

    static SeetaPointF operator+(const SeetaPointF &lhs, const SeetaPointF &rhs) {
		SeetaPointF result;
		result.x = lhs.x + rhs.x;
		result.y = lhs.y + rhs.y;

		return result;
    }

    static SeetaPointF operator-(const SeetaPointF &lhs, const SeetaPointF &rhs) {
		SeetaPointF result;
		result.x = lhs.x - rhs.x;
		result.y = lhs.y - rhs.y;

		return result;
    }

    static SeetaPointF operator/(const SeetaPointF &lhs, double rhs) {
		SeetaPointF result;
		result.x = lhs.x / rhs;
		result.y = lhs.y / rhs;

		return result;
    }

    static SeetaPointF operator*(const SeetaPointF &lhs, double rhs) {
		SeetaPointF result;
		result.x = lhs.x * rhs;
		result.y = lhs.y * rhs;

		return result;
    }

    static double operator^(const SeetaPointF &lhs, const SeetaPointF &rhs) {
        auto dx = lhs.x - rhs.x;
        auto dy = lhs.y - rhs.y;
        return std::sqrt(dx * dx + dy * dy);
    }

    /**
     * line for ax + by + c = 0
     */
    class Line {
    public:
        Line() = default;
        Line(double a, double b, double c)
            : a(a), b(b), c(c) {}

        Line(const SeetaPointF &a, const SeetaPointF &b) {
            auto x1 = a.x;
            auto y1 = a.y;
            auto x2 = b.x;
            auto y2 = b.y;
            // for (y2-y1)x-(x2-x1)y-x1(y2-y1)+y1(x2-x1)=0
            this->a = y2 - y1;
            this->b = x1 - x2;
            this->c = y1 * (x2 - x1) - x1 * (y2 - y1);
        }

        double distance(const SeetaPointF &p) const {
            return std::fabs(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
        }

        static bool near_zero(double f) {
            return f <= DBL_EPSILON && -f <= DBL_EPSILON;
        }

        SeetaPointF projection(const SeetaPointF &p) const {
             if (near_zero(a)) {
				 SeetaPointF result;
				 result.x = p.x;
				 result.y = -c / b;
				 return  result;
             }
            if (near_zero(b)) {
				SeetaPointF result;
				result.x = -c / a;
				result.y = p.y;
				return result;
            }
            // y = kx + b  <==>  ax + by + c = 0
            auto k = -a / b;
            SeetaPointF o = {0, -c / b};
            SeetaPointF project = {0};
            project.x = (float) ((p.x / k + p.y - o.y) / (1 / k + k));
            project.y = (float) (-1 / k * (project.x - p.x) + p.y);
            return project;
        }

        double a = 0;
        double b = 0;
        double c = 0;
    };

    static int quality_level(double score, float thresh0, float thresh1) {
        if (score < thresh0) {
            return 2;
        } else if (score < thresh1) {
            return 1;
        } else {
            return 0;
        }
    }

    QualityResult QualityOfPose::check(const SeetaImageData &image, const SeetaRect &face, const SeetaPointF *points,
                                       const int32_t N) {
        assert(points != nullptr && N == 5);
        auto &data = *reinterpret_cast<Data *>(m_data);

        auto point_center_eye = (points[0] + points[1]) / 2;
        auto point_center_mouth = (points[3] + points[4]) / 2;

        Line line_eye_mouth(point_center_eye, point_center_mouth);

        auto vector_left2right = points[1] - points[0];

        auto rad = atan2(vector_left2right.y, vector_left2right.x);
        auto angle = rad * 180 * M_PI;

        auto roll_dist = fabs(angle) / 180;

        auto raw_yaw_dist = line_eye_mouth.distance(points[2]);
        auto yaw_dist = raw_yaw_dist / (points[0] ^ points[1]);

        auto point_suppose_projection = point_center_eye * data.nose_center + point_center_mouth * (1 - data.nose_center);
        auto point_projection = line_eye_mouth.projection(points[2]);
        auto raw_pitch_dist = point_projection ^ point_suppose_projection;
        auto pitch_dist = raw_pitch_dist / (point_center_eye ^ point_center_mouth);

        int quality_roll = quality_level(roll_dist, data.roll0, data.roll1);
        int quality_yaw = quality_level(yaw_dist, data.yaw0, data.yaw1);
        int quality_pitch = quality_level(pitch_dist, data.pitch0, data.pitch1);

        int quality = std::min(std::min(quality_roll, quality_yaw), quality_pitch);
        float score = 3 - (roll_dist + yaw_dist + pitch_dist);

        return QualityResult(QualityLevel(quality), score);

        /*
        seeta::cv::ImageData cvimage = image;
        auto canvas = cvimage.toMat();

        cv::line(canvas,
                 cv::Point(points[0].x, points[0].y),
                 cv::Point(points[1].x, points[1].y),
                 CV_RGB(128, 128, 255), 2);

        cv::line(canvas,
                 cv::Point(points[3].x, points[3].y),
                 cv::Point(points[4].x, points[4].y),
                 CV_RGB(128, 128, 255), 2);

        cv::line(canvas,
                 cv::Point(point_center_eye.x, point_center_eye.y),
                 cv::Point(point_center_mouth.x, point_center_mouth.y),
                 CV_RGB(128, 128, 255), 2);

        cv::line(canvas,
                 cv::Point(point_projection.x, point_projection.y),
                 cv::Point(points[2].x, points[2].y),
                 CV_RGB(255, 128, 128), 2);

        for (int i = 0; i < N; ++i) {
            auto point = points[i];
            cv::circle(canvas, cv::Point(point.x, point.y), 3, CV_RGB(128, 255, 128), -1);
        }

        cv::circle(canvas, cv::Point(point_center_eye.x, point_center_eye.y), 3, CV_RGB(255, 255, 128), -1);
        cv::circle(canvas, cv::Point(point_center_mouth.x, point_center_mouth.y), 3, CV_RGB(255, 255, 128), -1);

        cv::circle(canvas, cv::Point(point_projection.x, point_projection.y), 3, CV_RGB(128, 255, 255), -1);

        ::cv::imshow("Example", canvas);
        ::cv::waitKey();

        return QualityResult(LOW, raw_yaw_dist);
         */
    }
}
