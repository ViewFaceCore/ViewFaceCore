//
// Created by seeta on 2018/6/29.
//

#include <global/operator_factory.h>
#include <iostream>

class Op : public ts::Operator {
public:
    Op() {
        std::cout << "Create Op" << std::endl;
    }

    virtual int run(ts::Stack &stack) {
        return 0;
    }
    virtual int infer(ts::Stack &stack, std::vector<ts::Tensor::Prototype> &output) {
        return 0;
    }
};

TS_REGISTER_OPERATOR(Op, ts::CPU, "op:cpu:float")

int main() {
    auto f = ts::OperatorCreator::Query(ts::CPU, "op:cpu:float");
    auto t = f();
    return 0;
}