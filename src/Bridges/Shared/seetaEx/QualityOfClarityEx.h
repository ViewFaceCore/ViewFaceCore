#pragma once
#include "seeta/QualityOfLBN.h"
#include "seeta/QualityStructure.h"
#include "seeta/FaceLandmarker.h"

using namespace std;
namespace seeta {

	class QualityOfClarityEx : public QualityRule
	{
	public:
		QualityOfClarityEx(std::string modelPath) {
			m_lbn = std::make_shared<QualityOfLBN>(ModelSetting(modelPath + "quality_lbn.csta"));
			m_marker = std::make_shared<FaceLandmarker>(ModelSetting(modelPath + "face_landmarker_pts68.csta"));
		}
		QualityOfClarityEx(float blur_thresh, std::string modelPath) {
			m_lbn = std::make_shared<QualityOfLBN>(ModelSetting(modelPath + "quality_lbn.csta"));
			m_marker = std::make_shared<FaceLandmarker>(ModelSetting(modelPath + "face_landmarker_pts68.csta"));
			m_lbn->set(QualityOfLBN::PROPERTY_BLUR_THRESH, blur_thresh);
		}
		QualityResult check(const SeetaImageData& image, const SeetaRect& face, const SeetaPointF* points, int32_t N) override {
			// assert(N == 68);
			auto points68 = m_marker->mark(image, face);
			int light, blur, noise;
			m_lbn->Detect(image, points68.data(), &light, &blur, &noise);
			if (blur == QualityOfLBN::BLUR) {
				return { QualityLevel::LOW, 0 };
			}
			else {
				return { QualityLevel::HIGH, 1 };
			}
		}
	private:
		std::shared_ptr<QualityOfLBN> m_lbn;
		std::shared_ptr<FaceLandmarker> m_marker;
	};
}

