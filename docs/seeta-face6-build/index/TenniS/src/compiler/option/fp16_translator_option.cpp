//
// Created by kier on 2019-06-10.
//

#include "compiler/option/fp16_translator_option.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"

#include "global/fp16_operator_factory.h"

bool
ts::Fp16TranslatorOption::translate(const ts::ComputingDevice &device,
                                    const ts::Node node,
                                    ts::Node &translated_node,
                                    const std::string &params,
                                    bool output_flag) const {

//    if (device.type() != GPU)
//        return false;

    auto op_name = node.bubble().op();

    if (Bubble::IsEndPoint(op_name)) {
        if (op_name == Bubble::Parameter)
            translated_node = bubble::param(node.bubble().name());
        else if (op_name == Bubble::Const) {
            translated_node = bubble::bubble(node.bubble());
        }
        return true;
    }

    if (op_name == name::layer::cast()) {
        translated_node = bubble::bubble(node.bubble());
        Node::Link(translated_node, node.inputs());
        return true;
    }

    if (op_name == name::layer::to_float()) {
        translated_node = bubble::bubble(node.bubble());
        Node::Link(translated_node, node.inputs());
        return true;
    }

    auto fp16_op_creator = Fp16OperatorCreator::Query(device.type(), node.bubble().op(), false);

    //status 0: current node is output but stream is fp16
    if (output_flag && fp16_op_creator != nullptr) {
        auto cast_fp32_node = bubble::op(node.bubble().name() + "_cast_fp32", name::layer::cast(), {node});
        Tensor dtype = tensor::from<int>(DTYPE::FLOAT32);
        cast_fp32_node.bubble().set(name::dtype, dtype);
        translated_node = cast_fp32_node;
        return true;
    }

    std::vector<Node> translated_inputs;
    bool const_to_fp16 = true;

    auto inputs = node.inputs();
    for (auto input : inputs) {
        //check current stream is fp16 or fp32
        auto input_creator = Fp16OperatorCreator::Query(device.type(), input.bubble().op(), false);
        bool stream_is_fp16 = false;
        if (input.bubble().op() == name::layer::cast()) {
            auto dtype = tensor::to_int(input.bubble().get(name::dtype));
            stream_is_fp16 = (dtype == FLOAT16) && input_creator != nullptr;
        } else if (Bubble::IsEndPoint(input.bubble().op())) {
            if (input.bubble().op() == Bubble::Const) {
                auto dtype = input.bubble().get(name::value).dtype();
                //only fp16,fp32,fp64 can convert to fp16
                if (dtype != FLOAT16 && dtype != FLOAT32 && dtype != FLOAT64) {
                    const_to_fp16 = false;
                }
                stream_is_fp16 = (dtype == FLOAT16);
            }
        } else {
            stream_is_fp16 = input_creator != nullptr;
        }

        //status: current stream is not fp16 but op support fp16 and to_fp16 is true
        if (!stream_is_fp16 && fp16_op_creator != nullptr && const_to_fp16) {
            auto cast_fp16_node = bubble::op(input.bubble().name() + "_cast_fp16", name::layer::cast(),
                                             {input});
            Tensor dtype = tensor::from<int>(DTYPE::FLOAT16);
            cast_fp16_node.bubble().set(name::dtype, dtype);
            translated_inputs.emplace_back(cast_fp16_node);
        }
            //status: current stream is fp16 but op doesn't support fp16
        else if (stream_is_fp16 && fp16_op_creator == nullptr) {
            auto cast_fp32_node = bubble::op(input.bubble().name() + "_cast_fp32", name::layer::cast(),
                                             {input});
            Tensor dtype = tensor::from<int>(DTYPE::FLOAT32);
            cast_fp32_node.bubble().set(name::dtype, dtype);
            translated_inputs.emplace_back(cast_fp32_node);
        } else {
            translated_inputs.emplace_back(input);
        }
    }

    translated_node = bubble::bubble(node.bubble());
    Node::Link(translated_node, translated_inputs);

    //NOTE:Advance calculations are used to improve accuracy on batch norm
    if (translated_node.bubble().op() == name::layer::batch_norm()) {
        bool change_param = false;
        auto epsilon = translated_node.bubble().get(name::epsilon);
        auto variance_node = translated_node.inputs()[2];
        DTYPE dtype = VOID;
        while (!Bubble::IsEndPoint(variance_node.bubble().op())) {
            variance_node = variance_node.inputs()[0];
            if (variance_node.bubble().op() == Bubble::Const) {
                dtype = variance_node.bubble().get(name::value).dtype();
                change_param = true;
            }
        }
        if (change_param) {
            Tensor variance = variance_node.bubble().get(name::value);
            Tensor epsilon_fp32 = tensor::cast(FLOAT32, epsilon);
            Tensor variance_fp32 = tensor::cast(FLOAT32, variance);
            auto epsilon_data = epsilon_fp32.data<float>();
            auto variance_data = variance_fp32.data<float>();

            for (int i = 0; i < variance.count(); i++) {
                variance_data[i] += *epsilon_data;
            }
            *epsilon_data = 0.f;

            Tensor epsilon_temp = tensor::cast(dtype, epsilon_fp32);
            Tensor variance_temp = tensor::cast(dtype, variance_fp32);
            auto test_e = epsilon_temp.data<float>();
            auto test_v = variance_temp.data<float>();
            TS_UNUSED(test_e);
            TS_UNUSED(test_v);
            translated_node.bubble().set(name::epsilon, epsilon_temp);
            variance_node.bubble().set(name::value, variance_temp);
        }
    }

    return true;

}


//TS_REGISTER_TRANSLATOR_OPTION(ts::Fp16TranslatorOption);
