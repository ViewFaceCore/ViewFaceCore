//
// Created by kier on 2019-07-24.
//

#include "seeta/QualityOfResolution.h"

namespace seeta {
	namespace {
		struct Data {
			/**
			* [0, resolution_low)=> LOW
			* [resolution_low, resolution_high)=> MEDIUM
			* [resolution_high, ~) => HIGH
			*/
			float resolution_low = 80.0f;
			float resolution_high = 128.0f;
		};
	}

		QualityOfResolution::QualityOfResolution() {
			m_data = new Data;
		}

		QualityOfResolution::QualityOfResolution(float low, float high) {
			m_data = new Data;
			Data* inner_data = reinterpret_cast<Data *>(m_data);
			inner_data->resolution_low = low;
			inner_data->resolution_high = high;
		}

		QualityOfResolution::~QualityOfResolution() {
			delete reinterpret_cast<Data *>(m_data);
		}

		float get_resolution_score(Data* inner_data, float resolution) 
		{
			return resolution;
		}
		QualityResult QualityOfResolution::check(
			const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF *points,
			const int32_t N) {
			float resolution = std::min(face.width, face.height);

			Data* inner_data = reinterpret_cast<Data *>(m_data);
			QualityLevel level;
			if (resolution < inner_data->resolution_low)
			{
				level = QualityLevel::LOW;
			}
			else if (resolution >= inner_data->resolution_low && resolution < inner_data->resolution_high)
			{
				level = QualityLevel::MEDIUM;
			}
			else if (resolution >= inner_data->resolution_high)
			{
				level = QualityLevel::HIGH;
			}

			return QualityResult(level, get_resolution_score(inner_data, resolution));
		}
}
