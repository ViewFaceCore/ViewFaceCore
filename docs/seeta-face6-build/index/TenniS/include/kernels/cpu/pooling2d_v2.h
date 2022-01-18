#ifndef TENSORSTACK_KERNELS_CPU_POOLING2D_V2_H
#define TENSORSTACK_KERNELS_CPU_POOLING2D_V2_H

#include "operator_on_cpu.h"
#include "backend/base/base_pooling2d_v2.h"
#include "pooling2d_core.h"


namespace ts {
	namespace cpu {
		class Pooling2DV2 : public OperatorOnAny<Operator> {
		public:
			using self = Pooling2DV2;
			using supper = OperatorOnAny<Operator>;

			Pooling2DV2();

			void init() override;

			/**
			 * Stack has: x, padding, ksize, stride
			 */
			int run(Stack &stack) override;

			int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

		private:
			Operator::shared m_op_pooling2d;
            // Conv2DFormat m_format;
            // Pooling2DType m_type;
            // Padding2DType m_padding_type;

            // std::valarray<int> m_padding4x2;
            // std::valarray<int> m_ksize4;
            // std::valarray<int> m_stride4;

			Tensor m_padding_int4x2;    // save pre set padding
			Tensor m_ksize_int4;
			Tensor m_stride_int4;
		};
	}
}


#endif //TENSORSTACK_KERNELS_CPU_POOLING2D_V2_H