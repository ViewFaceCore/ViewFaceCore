//
// Created by kier on 2019/1/19.
//

#include <backend/zoo/copy.h>

#include "backend/zoo/copy.h"
#include "runtime/stack.h"
#include "backend/name.h"
#include "global/operator_factory.h"

namespace ts {
    namespace zoo {

        void Copy::init() {
            supper::init();

            m_output_count = output_count();
        }

        int Copy::infer(Stack &stack, std::vector<Tensor::Prototype> &output) {
            // TS_AUTO_CHECK(stack.size() == 1);
            TS_AUTO_CHECK(int(stack.size()) == m_output_count);
            output.resize(stack.size());
            for (size_t i = 0; i < stack.size(); ++i) {
                output[i] = stack[i].proto();
            }
            return int(stack.size());
        }

        int Copy::run(Stack &stack) {
            // TS_AUTO_CHECK(stack.size() == 1);
            TS_AUTO_CHECK(int(stack.size()) == m_output_count);
            return int(stack.size());
        }
    }
}

using namespace ts::zoo;

TS_REGISTER_OPERATOR(Copy, ts::CPU, ts::name::layer::copy());
