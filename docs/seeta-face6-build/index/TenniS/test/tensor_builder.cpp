//
// Created by kier on 2018/10/30.
//

#include <iostream>
#include <core/tensor_builder.h>
#include <global/setup.h>
#include <core/tensor.h>

int main() {
    using namespace ts;
    setup();

    std::string str = "abcd";

    auto a = tensor::from(str);

    std::cout << str << " vs. " << tensor::to_string(a) << std::endl;

    Tensor ti(ts::FLOAT32, {1});
    ti.data<float>()[0] = 12.3f;
    Tensor tf = tensor::cast(ts::INT32, ti);

    std::cout << tf.data<int>()[0] << std::endl;

    Tensor tti = tensor::from(10.0f);
    std::cout << tensor::to_float(tti) << std::endl;

}
