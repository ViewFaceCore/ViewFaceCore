#pragma once
#include "../../../SeetaFace/index/build/include/seeta/QualityStructure.h"
#include "../../../SeetaFace/index/build/include/seeta/FaceLandmarker.h"
using namespace std;

namespace seeta {

	class QualityOfNoMask : public QualityRule
	{
	public:
		QualityOfNoMask(std::string modelPath) {
			m_marker = std::make_shared<seeta::FaceLandmarker>(ModelSetting(modelPath + "face_landmarker_mask_pts5.csta"));
		}
		QualityResult check(const SeetaImageData& image, const SeetaRect& face, const SeetaPointF* points, int32_t N) override {
			auto mask_points = m_marker->mark_v2(image, face);
			int mask_count = 0;
			for (auto point : mask_points) {
				if (point.mask) mask_count++;
			}
			QualityResult result;
			if (mask_count > 0) {
				return { QualityLevel::LOW, 1 - float(mask_count) / mask_points.size() };
			}
			else {
				return { QualityLevel::HIGH, 1 };
			}
		}
	private:
		std::shared_ptr<seeta::FaceLandmarker> m_marker;
	};
}

