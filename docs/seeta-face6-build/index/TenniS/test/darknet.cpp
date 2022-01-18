//
// Created by kier on 2019-05-28.
//

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "api/cpp/tensorstack.h"

int main() {
    using namespace ts::api;
    Device device("cpu", 0);

    cv::Mat cvimage = cv::imread("dog.jpg");
    std::string model = "yolov3.coco.tsm";
    std::vector<std::string> label = {
            "person", "bicycle", "car", "motorbike",
            "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "pottedplant", "bed", "diningtable", "toilet",
            "tvmonitor", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush",
    };

    auto WIDTH = 416;
    auto DEST = std::max(cvimage.rows, cvimage.cols);

    Workbench bench(device);
    bench.setup_context();
    bench.set_computing_thread_number(4);

    ImageFilter filter(device);
    filter.channel_swap({2, 1, 0});
    filter.to_float();
    filter.scale(1 / 255.0);
    filter.letterbox(WIDTH, WIDTH, 0.5);    // darknet pre-processor
    filter.to_chw();

    bench.setup(bench.compile(Module::Load(model)));
    bench.bind_filter(0, filter);

    Tensor tensor = tensor::build(UINT8, {1, cvimage.rows, cvimage.cols, cvimage.channels()}, cvimage.data);
    bench.input(0, tensor);

    bench.run();

    struct BBox {
        float x;
        float y;
        float w;
        float h;
        float scale;
        float label;
    };
    auto output_count = bench.output_count();

    std::stringstream oss;
    for (int i = 0; i < output_count; ++i) {
        auto output = bench.output(i);
        output = tensor::cast(FLOAT32, output);
        output = output.view(Tensor::InFlow::HOST);

        int N = output.size(0);

        std::cout << "N=" << N << std::endl;

        for (int n = 0; n < N; ++n) {
            auto box = output.data<BBox>(n);
            box.x *= float(DEST);
            box.y *= float(DEST);
            box.w *= float(DEST);
            box.h *= float(DEST);

            srand(int(box.label));
            auto r = rand() % 128 + 64;
            auto g = rand() % 128 + 64;
            auto b = rand() % 128 + 64;

            cv::rectangle(cvimage, cv::Rect(box.x, box.y, box.w, box.h), CV_RGB(r, g, b), 2);

            oss.str("");
            if (label.empty()) {
                oss << int(box.label) << ": " << int(std::round(box.scale) * 100) << "%";
            } else {
                oss << label[int(box.label)] << ": " << int(std::round(box.scale * 100)) << "%";
            }

            std::cout << oss.str() << std::endl;
            cv::putText(cvimage, oss.str(), cv::Point(box.x, box.y - 5), CV_FONT_HERSHEY_DUPLEX, 0.5, CV_RGB(r - 32, g - 32, b - 32));
        }
    }

    cv::imshow("YOLOv3", cvimage);
    cv::waitKey();

    return 0;
}