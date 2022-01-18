//
// Created by kier on 19-5-18.
//

#ifndef TENSORSTACK_DECLARE_OPERATOR_H
#define TENSORSTACK_DECLARE_OPERATOR_H

#include <core/tensor_builder.h>
#include "declaration.h"

#include "api/operator.h"
#include "global/operator_factory.h"

#include "declare_tensor.h"

#include "runtime/stack.h"

#include "core/device_context.h"
#include "runtime/runtime.h"

struct ts_OperatorParams {
public:
    ts_OperatorParams(ts::Operator *op) : op(op) {}

    ts::Operator *op;
};

class APIPluginStack {
public:
    APIPluginStack(ts::Stack &stack) {
        auto argc = stack.size();
        try {
            for (size_t i = 0; i < argc; ++i) {
                auto &tensor = stack[i];
                args.emplace_back(new ts_Tensor(tensor));
            }
        } catch (const ts::Exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        } catch (const std::exception &e) {
            for (auto &arg : args) delete arg;
            throw e;
        }
    }

    ~APIPluginStack() {
        for (auto &arg : args) {
            delete arg;
        }
    }

    std::vector<ts_Tensor *> args;
};


struct ts_OperatorContext {
public:
    ts_OperatorContext() {
        device = &ts::ctx::of<ts::DeviceContext>::ref();
        runtime = &ts::ctx::of<ts::RuntimeContext>::ref();
    }

    ts::DeviceContext *device;
    ts::RuntimeContext *runtime;
};

#endif //TENSORSTACK_DECLEARE_OPERATOR_H
