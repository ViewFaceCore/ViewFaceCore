#include "seeta/QualityOfIntegrity.h"

#include <orz/utils/log.h>
#include <orz/utils/except.h>
#include <algorithm>
namespace seeta {
	namespace {
		struct Data {
			/*
			 * shift_thresh image outside to inner's ratio
			 * face inner shift_thresh_high => HIGH
			 * face between shift_thresh_low and shift_thresh_high => MEDIUM
			 * face beyond shift_thresh_low pixel number => LOW
			 */
			float shift_thresh_low = 10.0;
			float shift_thresh_high = 1.5;
		};
	}

		QualityOfIntegrity::QualityOfIntegrity() {
			m_data = new Data;
		}

		QualityOfIntegrity::QualityOfIntegrity(float low, float high) {
			if (!(low >=0 && high >= 1.0))
			{
				orz::Log(orz::LogLevel::ERROR) << "input pamameters invalid: make sure low >=0 && high >= 1.0" << orz::crash;
			}

			m_data = new Data;
			auto &inner_data = *reinterpret_cast<Data *>(m_data);
			inner_data.shift_thresh_low = low;
			inner_data.shift_thresh_high = high;
		}

		float max(float a, float b)
		{
			return a > b ? a : b;
		}
		
		float min(float a, float b)
		{
			return a < b ? a : b;
		}

		QualityOfIntegrity::~QualityOfIntegrity() {
			delete reinterpret_cast<Data *>(m_data);
		}

		void get_range_by_pixel(const SeetaImageData &image, const SeetaRect &face,
			float pixels, SeetaPointF &left_top, SeetaPointF &right_bottom)
		{//get range
			left_top.x = face.x - pixels;
			left_top.y = face.y - pixels;

			right_bottom.x = face.x + face.width - 1 + pixels;
			right_bottom.y = face.y + face.height - 1 + pixels;

			return;
		}
		void get_range_by_ratio(const SeetaImageData &image, const SeetaRect &face,
						float ratio, SeetaPointF &left_top, SeetaPointF &right_bottom)
		{//get range
			float expand_half_time = (ratio - 1.0) / 2;
			left_top.x = face.x - face.width * expand_half_time;
			left_top.y = face.y - face.height * expand_half_time;

			right_bottom.x = face.x + face.width - 1 + face.width * expand_half_time;
			right_bottom.y = face.y + face.height - 1 + face.height * expand_half_time;

			return;
		}

		//float get_face_integrity(const SeetaImageData &image, const SeetaPointF &left_top, const SeetaPointF &right_bottom)
		//{//more center, more score
		//	SeetaPointF center_point;
		//	center_point.x = image.width / 2;
		//	center_point.y = image.height / 2;

		//	float x_max = max(abs(left_top.x - center_point.x), abs(left_top.y - center_point.y));
		//	float y_max = max(abs(right_bottom.x - center_point.x), abs(right_bottom.y - center_point.y));
		//	float distance = max(x_max, y_max);

		//	float score = 1.0 / (distance + 1);

		//	return score;
		//}

		QualityResult QualityOfIntegrity::check(
			const SeetaImageData &image,
			const SeetaRect &face,
			const SeetaPointF* points,
			const int32_t num)
		{
			auto &inner_data = *reinterpret_cast<Data *>(m_data);

			SeetaPointF left_top_low, right_bottom_low;
			SeetaPointF left_top_high, right_bottom_high;
			get_range_by_pixel(image, face, inner_data.shift_thresh_low, left_top_low, right_bottom_low);
			get_range_by_ratio(image, face, inner_data.shift_thresh_high, left_top_high, right_bottom_high);

			QualityLevel level;
			float score;
			if (left_top_low.x < 0 || left_top_low.y < 0 || 
				 (right_bottom_low.x >= image.width - 1) || 
				(right_bottom_low.y >= image.height - 1))
			{
				level = QualityLevel::LOW;
				score = 0.0;
			}
			else if (left_top_high.x >= 0 && left_top_high.y >= 0 &&
				(right_bottom_high.x <= image.width - 1) &&
				(right_bottom_high.y <= image.height - 1))
			{
				level = QualityLevel::HIGH;
				score = 1.0;
			}
			else
			{
				level = QualityLevel::MEDIUM;
				score = 0.5;
			}

			return QualityResult(level, score);
		}
}