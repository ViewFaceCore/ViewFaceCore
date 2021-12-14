//
// Created by kier on 19-7-24.
//

#ifndef SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H
#define SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H

#include "Struct.h"

namespace seeta {
	namespace v3 {
		/// <summary>
		/// 质量评估等级
		/// </summary>
		enum QualityLevel {
			/// <summary>
			/// 质量差
			/// </summary>
			LOW = 0,
			/// <summary>
			/// 质量一般
			/// </summary>
			MEDIUM = 1,
			/// <summary>
			/// 质量高
			/// </summary>
			HIGH = 2,
		};

		/// <summary>
		/// 质量评估结果
		/// </summary>
		class QualityResult {
		public:
			using self = QualityResult;
			/// <summary>
			/// 构造函数
			/// </summary>
			/// <returns></returns>
			QualityResult() = default;
			/// <summary>
			/// 
			/// </summary>
			/// <param name="level"></param>
			/// <param name="score"></param>
			/// <returns></returns>
			QualityResult(QualityLevel level, float score = 0) : level(level), score(score) {}

			/// <summary>
			/// 质量评估等级
			/// </summary>
			QualityLevel level = LOW;
			/// <summary>
			/// 质量评估分数
			/// <para>越大越好，没有范围限制</para>
			/// </summary>
			float score = 0;
		};

		struct QualityResultEx {
			int attr;
			QualityLevel level;   ///< quality level
			float score;          ///< greater means better, no range limit
		};

		struct QualityResultExArray {
			int size;
			QualityResultEx* data;
		};

		class QualityRule {
		public:
			using self = QualityRule;

			virtual ~QualityRule() = default;

			/**
			 * 开始评估
			 * @param image original image
			 * @param face face location
			 * @param points landmark on face
			 * @param N how many landmark on face given, normally 5
			 * @return Quality result
			 */
			virtual QualityResult check(
				const SeetaImageData& image,
				const SeetaRect& face,
				const SeetaPointF* points,
				int32_t N) = 0;
		};
	}
	using namespace v3;
}

#endif //SEETA_QULITY_EVALUATOR_QULITY_STRUCTURE_H
