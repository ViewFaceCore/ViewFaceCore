//
// Created by kier on 2019-04-15.
//

#ifndef TENSORSTACK_BACKEND_BASE_BASE_CHUNK_H
#define TENSORSTACK_BACKEND_BASE_BASE_CHUNK_H


#include "operator_on_device.h"

namespace ts {
    namespace base {
        class Chunk : public OperatorOnDevice {
        public:
            using self = Chunk;
            using supper = OperatorOnDevice;

            Chunk();

            void init() override;

            int run(Stack &stack) override;

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override;

            virtual void chunk(const Tensor &x, int chunks, int dim, std::vector<Tensor> &out) = 0;

        private:
            int m_chunks = 1;
            int m_dim = -2;
        };
    }
}


#endif //TENSORSTACK_BACKEND_BASE_BASE_CHUNK_H
