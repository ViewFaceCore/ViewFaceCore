#include "kernels/cpu/conv2d_algorithm.h"
#include <algorithm>

#include <module/graph.h>
#include <module/module.h>
#include <global/setup.h>
#include <runtime/workbench.h>
#include <global/operator_factory.h>
#include <utils/ctxmgr.h>
#include <core/tensor_builder.h>
#include <module/io/fstream.h>
#include <module/menu.h>
#include <core/tensor_builder.h>
#include "core/tensor_builder.h"

using namespace ts;

void test_speed(const std::string& device, int times, const Shape& input_shape,
    const Shape& kernel, const std::string& comment) {
    Graph g;
    ctx::bind<Graph> _graph(g);
    ComputingDevice cd(device, 0);

    auto input_x = bubble::param("input");
    auto input_weight = bubble::param("input_weight");
    auto conv2d_op = bubble::op("conv2d_op", "conv2d", { input_x,input_weight });

    ts::Shape format_shape = { 4 };
    Tensor format_param_type = ts::tensor::from("NCHW");
    conv2d_op.bubble().set("format", format_param_type);

    //auto input_param = tensor::build(FLOAT32, input.shape, input.ini_list);
    ts::Tensor input_param(FLOAT32, input_shape);
    for (int i = 0; i < input_param.count(); i++)
    {
        input_param.data<float>()[i] = i;
    }
    //auto kernel_param = tensor::build(FLOAT32, kernel.shape, kernel.ini_list);
    ts::Tensor input_weight_param(FLOAT32, kernel);
    for (int i = 0; i < input_weight_param.count(); i++)
    {
        input_weight_param.data<float>()[i] = i;
    }

    ts::Shape padding_shape = { 4,2 };
    ts::Shape stride_shape = { 4 };
    ts::Shape dilation_shape = { 4 };
    auto stride_param = tensor::build(FLOAT32, stride_shape, { 1,1,1,1 });
    auto dilation_param = tensor::build(FLOAT32, dilation_shape, { 1,1,1,1 });
    auto pad_param = tensor::build(FLOAT32, padding_shape, { 0,0,0,0,0,0,0,0 });

    conv2d_op.bubble().set("padding", pad_param);
    conv2d_op.bubble().set("stride", stride_param);
    conv2d_op.bubble().set("dialations", dilation_param);


    // setup module
    std::shared_ptr<Module> m = std::make_shared<Module>();
    m->load(g, { "conv2d_op" });

    Workbench::shared bench;

    try {
        bench = Workbench::Load(m, cd);
        bench = bench->clone();
    }
    catch (const Exception &e) {
        std::cout << e.what() << std::endl;
        return;
    }

    int num_count = times;
    bench->do_profile(true);
    bench->input("input", input_param);
    bench->input("input_weight", input_weight_param);
    for (size_t i = 0; i < num_count; i++)
    {
        //time_log _log(ts::LOG_INFO, "Spent ");
        bench->run();
    }
    std::cout << "====device: " << device << " comment: " << comment << "====" << std::endl;
    bench->profiler().log(std::cout);
    auto output_c = bench->output("conv2d_op");
//    std::vector<int> vec = output_c.sizes();
    std::vector<int> vec = (const std::vector<int> &) output_c.sizes();
    std::cout << "output size: ";
    for (int i = 0; i<vec.size(); i++)
        std::cout << vec[i] << ",";
    std::cout << std::endl;
}

using test_function = std::function<void(const Tensor&, const Tensor &, Tensor &)>;

void print_avg_time(const std::string &title, const int times, const Shape& input_shape,const Shape& kernel_shape,
    test_function func,bool f23_kernel_flag) {

    Tensor kernel(FLOAT32, kernel_shape);
    float* kernel_data = kernel.data<float>();
    for (int i = 0; i < kernel.count(); i++)
    {
        kernel_data[i] = 1.0f;
    }
    Shape kernel_trans_shape;
    kernel_trans_shape.resize(4);
    kernel_trans_shape[0] = kernel_shape[0];
    kernel_trans_shape[1] = kernel_shape[1];
    if (f23_kernel_flag){
        kernel_trans_shape[2] = 4;
        kernel_trans_shape[3] = 4;
    }
    else {
        kernel_trans_shape[2] = 8;
        kernel_trans_shape[3] = 8;
    }
    Tensor kernel_trans(FLOAT32, kernel_trans_shape);
    if(f23_kernel_flag)
        cpu::Conv2dAlgorithm<float>::conv3x3_winograd23_transform_kernel(kernel, kernel_trans);
    else
        cpu::Conv2dAlgorithm<float>::conv3x3_winograd63_transform_kernel(kernel, kernel_trans);

    Tensor x(FLOAT32, input_shape);
    float* x_ptr = x.data<float>();
    for (int i = 0; i < x.count(); i++)
    {
        x_ptr[i] = i;
    }

    Shape out_shape = { input_shape[0],kernel_shape[0],input_shape[2]-2,input_shape[3]-2 };
    Tensor out(FLOAT32, out_shape);

    using namespace std::chrono;
    microseconds duration(0);

    auto start = system_clock::now();

    for (int i = 0; i < times; ++i) {
        func(x, kernel_trans, out);
    }

    auto end = system_clock::now();
    duration += duration_cast<microseconds>(end - start);
    double spent = 1.0 * duration.count() / 1000;
    std::cout << title << ", spent=" << spent << "ms" << " avg time = " << spent /times << "ms" << std::endl;
}

int main()
{

    ts::RuntimeContext runtime;
    runtime.set_computing_thread_number(1);

    ts::ctx::bind<ts::RuntimeContext> _bind_runtime(runtime);

    //Shape kernel_shape = { 2,2,3,3 };
    //Tensor kernel = tensor::build(FLOAT32, kernel_shape, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 });
    int times = 1000;

    Shape kernel_shape = { 2,256,3,3 };
    Shape x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 2,64,3,3 }");

    kernel_shape = { 4,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 4,64,3,3 }");

    kernel_shape = { 8,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 8,64,3,3 }");

    kernel_shape = { 16,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 16,64,3,3 }");

    kernel_shape = { 32,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 32,64,3,3 }");

    kernel_shape = { 64,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 64,64,3,3 }");

    kernel_shape = { 128,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 128,64,3,3 }");

    kernel_shape = { 256,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 256,64,3,3 }");

    kernel_shape = { 512,256,3,3 };
    x_shape = { 1,256,21,21 };
    print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd23,true);
    print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, cpu::Conv2dAlgorithm<float>::conv3x3_winograd63,false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 512,64,3,3 }");

    //test input size
    //Shape kernel_shape = { 64,64,3,3 };
    //Shape x_shape = { 1,64,21,21 };
    //print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd23, true);
    //print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd63, false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 64,64,3,3 }");
    //test_speed("opt", times, x_shape, kernel_shape, "input:{ 1,64,21,21 },kernel:{ 64,64,3,3 }");

    //kernel_shape = { 64,64,3,3 };
    //x_shape = { 1,64,8,8 };
    //print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd23, true);
    //print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd63, false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,8,8 },kernel:{ 64,64,3,3 }");
    //test_speed("opt", times, x_shape, kernel_shape, "input:{ 1,64,8,8 },kernel:{ 64,64,3,3 }");

    //kernel_shape = { 64,64,3,3 };
    //x_shape = { 1,64,50,50 };
    //print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd23, true);
    //print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd63, false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,50,50 },kernel:{ 64,64,3,3 }");
    //test_speed("opt", times, x_shape, kernel_shape, "input:{ 1,64,50,50 },kernel:{ 64,64,3,3 }");

    //kernel_shape = { 64,64,3,3 };
    //x_shape = { 1,64,100,100 };
    //print_avg_time("conv3x3_winograd23", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd23, true);
    //print_avg_time("conv3x3_winograd63", times, x_shape, kernel_shape, opt::Conv2dAlgorithm<float>::conv3x3_winograd63, false);
    //test_speed("cpu", times, x_shape, kernel_shape, "input:{ 1,64,100,100 },kernel:{ 64,64,3,3 }");
    //test_speed("opt", times, x_shape, kernel_shape, "input:{ 1,64,100,1000 },kernel:{ 64,64,3,3 }");



#ifdef _OUT_DEBUG
    float* out_ptr = out.data<float>();
    float* out_at = out_ptr;

    std::cout << "output: " << std::endl;
    for (int n = 0; n < out_shape[0]; n++)
    {
        for (int c = 0; c < out_shape[1]; c++)
        {
            for (int h = 0; h < out_shape[2]; h++)
            {
                for (int w = 0; w < out_shape[3]; w++)
                {
                    std::cout << *out_at++ << " ";
                }
                std::cout << std::endl;
            }
        }
    }
#endif

    return 0;
}