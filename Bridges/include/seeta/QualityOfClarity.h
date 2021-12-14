#ifndef SEETA_QUALITYEVALUATOR_QUALITYOFCLARITY_H
#define SEETA_QUALITYEVALUATOR_QUALITYOFCLARITY_H
#include "QualityStructure.h"

namespace seeta {
	namespace v3 {
		class QualityOfClarity : public QualityRule {
		public:
			using self = QualityOfClarity;
			using supper = QualityRule;

			/**
			 * Construct with recommanded parameters
			 */
			SEETA_API QualityOfClarity();

			/*
			 *@param low
			 *@param high
			 *[0, low)=> LOW
			 *[low, high)=> MEDIUM
			 *[high, ~)=> HIGH
			 */
			SEETA_API QualityOfClarity(float low, float high);

			SEETA_API ~QualityOfClarity() override;

			SEETA_API QualityResult check(
				const SeetaImageData &image,
				const SeetaRect &face,
				const SeetaPointF* points,
				const int32_t num) override;

		private:
			QualityOfClarity(const QualityOfClarity &) = delete;
			QualityOfClarity &operator=(const QualityOfClarity &) = delete;
		private:
			void* m_data;

		};
	}
	using namespace v3;
}
#endif