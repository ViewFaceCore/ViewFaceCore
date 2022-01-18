#include "compiler/option/pack_translator_option.h"

#include "backend/name.h"
#include "core/tensor_builder.h"
#include "module/menu.h"
#include "kernels/cpu/math_cpu.h"
#include "kernels/common/math.h"

#include "global/operator_factory.h"
#include "kernels/common/function.h"
#include "compiler/argparse.h"

static bool has_defined_op(const ts::ComputingDevice &device, const std::string &op) {
    auto creator = ts::OperatorCreator::Query(device.type(), op, true);
    return creator != nullptr;
}

bool ts::PackTranslatorOption::translate(const ComputingDevice &device, const Node node,
    Node &translated_node, const std::string &params, bool output_flag) const {
    auto op_name = node.bubble().op();

    if (Bubble::IsEndPoint(op_name)) {
        if (op_name == Bubble::Parameter)
            translated_node = bubble::param(node.bubble().name());
        else if (op_name == Bubble::Const) {
            translated_node = bubble::bubble(node.bubble());
        }
        return true;
    }

    translated_node = bubble::bubble(node.bubble());

    if (op_name != name::layer::conv2d() && op_name != name::layer::conv2d_v2()
        && op_name != name::layer::inner_prod() && op_name != name::layer::gemm()) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    // not translate node if non-cpu device has defined op
    if (device.type() != "cpu" && has_defined_op(device, op_name)) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    //add gemm translate support,to alpha*[op(A)*op(B)]+beta*C
    if (op_name == name::layer::gemm()) {
        auto name = node.bubble().name();
        auto inputs = node.inputs();
        auto A_node = inputs[0],B_node = inputs[1],C_node = inputs[2];

        float alpha = tensor::to_float(node.bubble().get(name::alpha));
        float beta = tensor::to_float(node.bubble().get(name::beta));
        bool transA = tensor::to_bool(node.bubble().get(name::transA));
        bool transB = tensor::to_bool(node.bubble().get(name::transB));

        //NOTE:Can not translate in this case,inner_product not support now
        if (A_node.bubble().op() != Bubble::Const && transA) {
            Node::Link(translated_node, node.inputs());
            return true;
        }

        //transpose const if trans is true
        auto A_trans_node = A_node;
        auto B_trans_node = B_node;
        if (A_node.bubble().op() == Bubble::Const && transA) {
            auto A_tensor = A_node.bubble().get(name::value);
            auto dtype = A_tensor.dtype();
            auto shape = A_tensor.sizes();
            Shape transposed_shape({ shape[1], shape[0] });
            Tensor transposed(dtype, transposed_shape);
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { cpu::math<TYPE, TYPE>::matrix_transpose(A_tensor.data<TYPE>(), transposed.data<TYPE>(), shape[0], shape[1]); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Pack translator not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
            A_trans_node.bubble().set(name::value, transposed);
        }

        if (B_node.bubble().op() == Bubble::Const && transB) {
            auto B_tensor = B_node.bubble().get(name::value);
            auto dtype = B_tensor.dtype();
            auto shape = B_tensor.sizes();
            Shape transposed_shape({ shape[1], shape[0] });
            Tensor transposed(dtype, transposed_shape);
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { cpu::math<TYPE, TYPE>::matrix_transpose(B_tensor.data<TYPE>(), transposed.data<TYPE>(), shape[0], shape[1]); break; }
                DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << "Pack translator not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
            B_trans_node.bubble().set(name::value, transposed);
        }

        auto inner_prod_node = bubble::op(name + "_inner_prod", name::layer::inner_prod(), { A_trans_node, B_trans_node });

        auto alpha_node = bubble::data(name + "_alpha", node.bubble().get(name::alpha));
        auto alpha_mul_node = bubble::op(name + "_alpha_mul", name::layer::mul(), { alpha_node, inner_prod_node });

        auto beta_node = bubble::data(name + "_beta", node.bubble().get(name::beta));
        auto beta_mul_node = bubble::op(name + "_beta_mul", name::layer::mul(), { beta_node, C_node });

        if (ts::near(beta, float(0))) {
            if (ts::near(alpha, float(1)))
                translated_node = inner_prod_node;
            else
                translated_node = alpha_mul_node;
        }
        else {
            if (ts::near(alpha, float(1))) {
                auto add_node = bubble::op(name + "_add", name::layer::add(), { inner_prod_node, beta_mul_node });
                translated_node = add_node;
            }
            else {
                auto add_node = bubble::op(name + "_add", name::layer::add(), { alpha_mul_node, beta_mul_node });
                translated_node = add_node;
            }
        }
        translated_node->name(name);
        return true;
    }

    auto inputs = node.inputs();
    auto kernel_node = inputs[1];
    if (op_name == name::layer::conv2d() ||op_name == name::layer::inner_prod()) {
        kernel_node = inputs[1];
    }
    else if (op_name == name::layer::conv2d_v2()) {
        kernel_node = inputs[2];
    }

    if (kernel_node->op() != Bubble::Const) {
        Node::Link(translated_node, node.inputs());
        return true;
    }

    auto kernel_tensor = kernel_node.bubble().get(name::value);
    auto kernel_shape = kernel_tensor.sizes();
    auto kernel_type = kernel_tensor.dtype();

    //winograd_check
#ifdef TS_ON_ARM
    ArgParser parser;
    parser.add({"--winograd", "-win"}, {"--no-winograd", "-no-win"}, true);
    parser.parse(params);
    if (parser.get("--winograd")) {
        if(op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()){
            Tensor stride_tensor = tensor::cast(INT32, node.bubble().get(name::stride));
            Stride2D stride_size(stride_tensor.data<int>()[2], stride_tensor.data<int>()[3]);
            Tensor dilation_tensor = tensor::cast(INT32, node.bubble().get(name::dilation));
            Dilation2D dilation_size(dilation_tensor.data<int>()[2], dilation_tensor.data<int>()[3]);
            bool winograd_flag = false;
            winograd_flag = KernelCommonFunc<float>::winograd_check(kernel_shape, stride_size, dilation_size);
            if(winograd_flag){
                Node::Link(translated_node, node.inputs());
                return true;
            }
        }
    }
#endif

    int kernel_size_width;
    int kernel_size_height;

    if (op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()) {
        kernel_size_height = kernel_shape[0];
        kernel_size_width = kernel_shape[1] * kernel_shape[2] * kernel_shape[3];
    }
    else {
        kernel_size_height = kernel_shape[0];
        kernel_size_width = kernel_shape[1];
    }


    bool need_transpose = false;
    if (node.bubble().has("transpose")) {
        need_transpose = tensor::to_bool(node.bubble().get("transpose"));
    }

    Tensor kernel_packed(kernel_type, kernel_shape);

    if (op_name == name::layer::conv2d() || op_name == name::layer::conv2d_v2()) {
        switch (kernel_type) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { cpu::math<TYPE, TYPE>::pack8_A(kernel_size_height, kernel_size_width, kernel_tensor.data<TYPE>(), kernel_size_width, kernel_packed.data<TYPE>()); break; }
            DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Pack translator not support data type(" << kernel_type << "): " << type_str(kernel_type) << eject;
                break;
            }
        }
    }
    else {
        switch (kernel_type) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
            case DTYPE: { \
                if(need_transpose){ \
                    Shape transposed_shape({kernel_size_width, kernel_size_height}); \
                    Tensor transposed(kernel_type, transposed_shape); \
                    Tensor packed_transposed(kernel_type, transposed_shape); \
                    cpu::math<TYPE, TYPE>::matrix_transpose(kernel_tensor.data<TYPE>(), transposed.data<TYPE>(), kernel_size_height, kernel_size_width); \
                    cpu::math<TYPE, TYPE>::pack8_B(kernel_size_width, kernel_size_height, transposed.data<TYPE>(), kernel_size_height, packed_transposed.data<TYPE>()); \
                    kernel_packed = packed_transposed; \
                } \
                else{ \
                    cpu::math<TYPE, TYPE>::pack8_B(kernel_size_height, kernel_size_width, kernel_tensor.data<TYPE>(), kernel_size_width, kernel_packed.data<TYPE>()); \
                } break;}         
            DECLARE_COMPUTE_RUN(FLOAT32, float);
#undef DECLARE_COMPUTE_RUN
            default: {
                TS_LOG_ERROR << "Pack translator not support data type(" << kernel_type << "): " << type_str(kernel_type) << eject;
                break;
            }
        }
    }


    Node kernel_packed_node = kernel_node;
    kernel_packed_node.bubble().set(name::value, kernel_packed);
    translated_node.bubble().set(name::kernel_packed, tensor::from<bool>(true));

    if (op_name == name::layer::inner_prod()) {
        translated_node.bubble().set("transpose", tensor::from<bool>(false));
    }

    if(op_name == name::layer::conv2d() || op_name == name::layer::inner_prod())
        Node::Link(translated_node, { inputs[0], kernel_packed_node });
    else
        Node::Link(translated_node, { inputs[0], inputs[1], kernel_packed_node });

    return true;
}


