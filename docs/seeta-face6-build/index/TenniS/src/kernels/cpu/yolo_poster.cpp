//
// Created by kier on 2019-05-29.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"
#include <algorithm>

#include <cstring>
#include <cstdlib>

#ifdef _MSC_VER
#pragma warning(disable:4200)
#endif

namespace ts {
    namespace cpu {
        class YoloPoster : public Operator {
        public:
            using self = YoloPoster;
            using supper = Operator;

            YoloPoster() {
                field("thresh", REQUIRED); // object thresh
                field("nms", REQUIRED);   // nms thresh
            }

            void init() override {
                supper::init();

                m_thresh = tensor::to_float(get("thresh"));
                m_nms = tensor::to_float(get("nms"));
            }

            struct layer {
            public:
                layer() = default;
                layer(Tensor &x, int classes, int n) {
                    this->n = n;
                    this->h = x.size(2);
                    this->w = x.size(3);
                    this->output = x.data<float>();
                    this->outputs = x.size(1) * x.size(2) * x.size(3);
                    this->classes = classes;
                }
                int n;
                int h;
                int w;
                float *output;
                int outputs;
                int classes;
                float *biases;
                int *mask;
            };

            static int entry_index(layer l, int batch, int location, int entry) {
                int n = location / (l.w * l.h);
                int loc = location % (l.w * l.h);
                return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
            }

            static int yolo_num_detections(layer l, float thresh)
            {
                int i, n;
                int count = 0;
                for (i = 0; i < l.w*l.h; ++i){
                    for(n = 0; n < l.n; ++n){
                        int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
                        if(l.output[obj_index] > thresh){
                            ++count;
                        }
                    }
                }
                return count;
            }

            /**
             *
             * @param stack list of packed4tensor
             * @param output shape before nms
             * @return not used
             */
            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override {
                // TS_AUTO_CHECK(stack.size() == 1);

                auto num = stack.size();

                int out_classes = stack[-1].field(1).data<int32_t>(0);
                int out_n = stack[-1].field(0).size(0);

                int s = 0;
                for (int i = 1; i < num; ++i) {
                    auto features = stack[i].unpack();
                    auto yolo = tensor::cast(FLOAT32, features[0].view(MemoryDevice(CPU)));
                    auto &classes = features[1];
                    auto &mask = features[2];
                    // auto &anchors = features[3];

                    if (out_classes != classes.data<int32_t>(0)) {
                        TS_LOG_ERROR << "Input yolo classes mismatch." << eject;
                    }

                    layer l(yolo, out_classes, mask.count());
                    s += yolo_num_detections(l, m_thresh);
                }

                // output shape is 4(box: x, y, w, h) + 1(score) + 1(label)
                output.resize(out_n, Tensor::Prototype(FLOAT32, {s, 4 + 1 + 1}));
                // output number is batch size

                return 1;
            }

            struct detection {
                struct box {
                    float x;
                    float y;
                    float w;
                    float h;
                } bbox;
                float objectness;
                float prob[0];
            };

            using box = detection::box;

            class detection_list {
            public:
                detection_list(size_t width, size_t capacity = 128)
                    : m_width(width), m_size(0), m_capacity(capacity) {
                    m_data = std::make_shared<HardMemory>(MemoryDevice(CPU), m_capacity * m_width);
                }

                void clear() { m_size = 0; }

                size_t width() const { return m_width; }

                detection &append() {
                    ++m_size;
                    adjust();
                    return this->operator[](m_size - 1);
                }

                detection &operator[](size_t i) { return *reinterpret_cast<detection*>(data(i)); }

                const detection &operator[](size_t i) const { return *reinterpret_cast<const detection*>(data(i)); }

                detection &operator[](int i) { return *reinterpret_cast<detection*>(data(i)); }

                const detection &operator[](int i) const { return *reinterpret_cast<const detection*>(data(i)); }

                size_t size() const { return m_size; }

            private:
                void adjust() {
                    if (m_size > m_capacity) {
                        m_capacity = size_t(m_capacity * 2);
                        m_data->expect(m_capacity * m_width);
                    }
                }

                size_t index(int i) const { return i >= 0 ? size_t(i) : m_size - size_t(-i); }

                void *data(size_t i) { return m_data->data<char>() + i * m_width; }

                const void *data(size_t i) const { return m_data->data<char>() + i * m_width; }

                void *data(int i) { return m_data->data<char>() + index(i) * m_width; }

                const void *data(int i) const { return m_data->data<char>() + index(i) * m_width; }

                size_t m_width = 0;
                size_t m_size = 0;
                size_t m_capacity = 0;
                HardMemory::shared m_data;
            };

            static box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
            {
                box b;
                b.x = (i + x[index + 0*stride]) / lw;
                b.y = (j + x[index + 1*stride]) / lh;
                b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
                b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
                return b;
            }

            static void get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection_list &dets)
            {
                int i,j,n;
                float *predictions = l.output;
                // int count = 0;
                for (i = 0; i < l.w*l.h; ++i){
                    int row = i / l.w;
                    int col = i % l.w;
                    for(n = 0; n < l.n; ++n){
                        int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
                        float objectness = predictions[obj_index];
                        if(objectness <= thresh) continue;
                        int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
                        auto &this_detection = dets.append();
                        this_detection.bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
                        this_detection.objectness = objectness;
                        // dets[count].classes = l.classes;
                        for(j = 0; j < l.classes; ++j){
                            int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                            float prob = objectness*predictions[class_index];
                            this_detection.prob[j] = (prob > thresh) ? prob : 0;
                        }
                        // ++count;
                    }
                }
                // correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
                // return count;
            }

            static float overlap(float x1, float w1, float x2, float w2)
            {
                float l1 = x1 - w1/2;
                float l2 = x2 - w2/2;
                float left = l1 > l2 ? l1 : l2;
                float r1 = x1 + w1/2;
                float r2 = x2 + w2/2;
                float right = r1 < r2 ? r1 : r2;
                return right - left;
            }

            static float box_intersection(box a, box b)
            {
                float w = overlap(a.x, a.w, b.x, b.w);
                float h = overlap(a.y, a.h, b.y, b.h);
                if(w < 0 || h < 0) return 0;
                float area = w*h;
                return area;
            }

            static float box_union(box a, box b)
            {
                float i = box_intersection(a, b);
                float u = a.w*a.h + b.w*b.h - i;
                return u;
            }

            static float box_iou(box a, box b)
            {
                return box_intersection(a, b)/box_union(a, b);
            }

            static void do_nms_sort(detection_list &dets, int total, int classes, float thresh)
            {
                int i, j, k;

                std::vector<int> sorted_template(total);
                for (i = 0; i < total; ++i) {
                    sorted_template[i] = i;
                }

                for(k = 0; k < classes; ++k){
                    int sort_class = k;
                    auto sorted = sorted_template;
                    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) -> bool {
                        return dets[a].prob[sort_class] > dets[b].prob[sort_class];
                    });
                    for(i = 0; i < total; ++i){
                        auto sorted_i = sorted[i];
                        if(dets[sorted_i].prob[k] == 0) continue;
                        box a = dets[sorted_i].bbox;
                        for(j = i+1; j < total; ++j){
                            auto sorted_j = sorted[j];
                            box b = dets[sorted_j].bbox;
                            if (box_iou(a, b) > thresh){
                                dets[sorted_j].prob[k] = 0;
                            }
                        }
                    }
                }
            }

            static float clamp(float x, float min, float max) {
                return std::max(min, std::min(max, x));
            }

            static box clamp(box x, float min, float max) {
                auto left = clamp(x.x - x.w / 2, min, max);
                auto right = clamp(x.x + x.w / 2, min, max);
                auto top = clamp(x.y - x.h / 2, min, max);
                auto bottom = clamp(x.y + x.h / 2, min, max);
                auto width = right - left;
                auto height = bottom - top;
                box y;
                y.x = left + width / 2;
                y.y = top + height / 2;
                y.w = width;
                y.h = height;
                return y;
            }

            struct BindingBox {
                BindingBox() = default;
                BindingBox(float x, float y, float w, float h, float score, float label)
                        : x(x), y(y), w(w), h(h), score(score), label(label) {}
                BindingBox(float x, float y, float w, float h, float score, int label)
                        : x(x), y(y), w(w), h(h), score(score), label(float(label)) {}

                float x;
                float y;
                float w;
                float h;
                float score;
                float label;
            };

            int run(Stack &stack) override {
                auto num = stack.size();

                int out_classes = stack[-1].field(1).data<int32_t>(0);
                int out_n = stack[-1].field(0).size(0);

                int object_width = 4 + 1 + out_classes;
                detection_list dets(object_width * 4);

                std::vector<Tensor> yolo_list;
                for (int i = 1; i < num; ++i) {
                    auto yolo = tensor::cast(FLOAT32, stack[i].field(0).view(MemoryDevice(CPU)));
                    yolo_list.emplace_back(yolo);
                }

                auto &net = stack[0];
                auto neth = net.size(2);
                auto netw = net.size(3);

                std::vector<Tensor> result;

                for (int n = 0; n < out_n; ++n) {
                    dets.clear();
                    for (int i = 1; i < num; ++i) {
                        auto features = stack[i].unpack();
                        auto yolo = yolo_list[i - 1].slice(n, n + 1);   // got n-th feature map
                        auto classes = stack[i].field(1);
                        auto mask = stack[i].field(2);
                        auto anchors = stack[i].field(3);

                        if (out_classes != classes.data<int32_t>(0)) {
                            TS_LOG_ERROR << "Input yolo classes mismatch." << eject;
                        }

                        // construct layer
                        layer l(yolo, out_classes, mask.count());
                        l.biases = anchors.data<float>();
                        l.mask = mask.data<int32_t>();

                        get_yolo_detections(l, 0/* original image width */, 0, netw, neth, m_thresh, nullptr, 1, dets);
                    }

                    // clamp before nms
                    for (size_t i = 0; i < dets.size(); ++i) {
                        auto &det = dets[i];
                        det.bbox = clamp(det.bbox, 0, 1);
                    }

                    // do nms
                    do_nms_sort(dets, int(dets.size()), out_classes, m_nms);

                    // convert showing mode
                    std::vector<BindingBox> boxes;
                    for (size_t i = 0; i < dets.size(); ++i) {
                        auto &det = dets[i];
                        for (int j = 0; j < out_classes; ++j) {
                            if (det.prob[j] > m_thresh) {
                                boxes.emplace_back(det.bbox.x - det.bbox.w / 2, det.bbox.y - det.bbox.h / 2,
                                        det.bbox.w, det.bbox.h, det.prob[j], j);
                            }
                        }
                    }

                    // to tensor
                    Tensor field(Tensor::InFlow::HOST, FLOAT32, {int(boxes.size()), 6});
                    std::memcpy(field.data(), boxes.data(), field.count() * field.proto().type_bytes());

                    result.emplace_back(std::move(field));
                }

                Tensor out;
                out.pack(result);

                stack.push(out);

                return 1;
            }

        private:
            float m_thresh = 0.5f;
            float m_nms = 0.45f;
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(YoloPoster, CPU, name::layer::yolo_poster())
