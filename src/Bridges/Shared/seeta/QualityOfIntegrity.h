#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFINTEGRITY_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFINTEGRITY_H
#include "QualityStructure.h"

namespace seeta {
	namespace v3 {
		class QualityOfIntegrity : public QualityRule {
		public:
			using self = QualityOfIntegrity;
			using supper = QualityRule;

			SEETA_API QualityOfIntegrity();

			/*
			* shift_thresh image outside to inner's ratio
			* face inner high => HIGH
			* face between low and high => MEDIUM
			* face beyond low => LOW
			*/
			SEETA_API QualityOfIntegrity(float low, float high);

			SEETA_API ~QualityOfIntegrity() override;

			SEETA_API QualityResult check(
				const SeetaImageData &image,
				const SeetaRect &face,
				const SeetaPointF* points,
				const int32_t num) override;

		private:
			QualityOfIntegrity(const QualityOfIntegrity &) = delete;
			QualityOfIntegrity &operator=(const QualityOfIntegrity &) = delete;
		private:
			void* m_data;

		};
	}
	using namespace v3;
}
#endif