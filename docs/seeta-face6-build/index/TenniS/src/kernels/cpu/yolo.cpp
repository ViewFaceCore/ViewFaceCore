//
// Created by kier on 2019-05-29.
//

#include "runtime/operator.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include "core/tensor_builder.h"
#include "runtime/stack.h"

namespace ts {
    namespace cpu {
		template <typename T>
		static T neg(T a) { return -a; }

		template <> uint8_t neg(uint8_t a) { return a; }
		template <> uint16_t neg(uint16_t a) { return a; }
		template <> uint32_t neg(uint32_t a) { return a; }
		template <> uint64_t neg(uint64_t a) { return a; }

        class Yolo : public Operator {
        public:
            using self = Yolo;
            using supper = Operator;

            Yolo() {
                field("classes", REQUIRED); // number of classes
                field("mask", REQUIRED);   // mask
                field("anchors", REQUIRED); // anchors
            }

            void init() override {
                supper::init();

                param.classes = tensor::cast(INT32, get("classes"));
                param.mask = tensor::cast(INT32, get("mask"));
                param.anchors = tensor::cast(FLOAT32, get("anchors"));

                m_classes = tensor::to_int(param.classes);
                m_mask = tensor::array::to_int(param.mask);
                m_anchors = tensor::array::to_float(param.anchors);
                m_n = int(m_mask.size());
            }

            /**
             *
             * @param stack [x] {batch, C, H, W}
             * @param output {batch, self.n * (self.classes + 4 + 1), H, W}
             * @return not used
             */
            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) override {
                TS_AUTO_CHECK(stack.size() == 1);

                auto &x = stack[0];

                TS_AUTO_CHECK(x.dims() == 4);
                auto batch = x.size(0);
                // auto c = x.size(1);
                auto h = x.size(2);
                auto w = x.size(3);

                output.resize(4);
                output[0] = Tensor::Prototype(x.dtype(), {batch, m_n * (m_classes + 4 + 1), h, w});
                output[1] = param.classes.proto();
                output[2] = param.mask.proto();
                output[3] = param.anchors.proto();

                if (x.size(1) != output[0].size(1)) {
                    TS_LOG_ERROR << "Input and output channels mismatch, got "
                                 << x.size(1) << " vs. " << output[0].size(1) << eject;
                }

                return 1;
            }

            struct layer {
                int w;
                int h;
                int outputs;
                int classes;
                int batch;
                int n;
                void *output;
            };

            static int entry_index(layer l, int batch, int location, int entry) {
                int n = location / (l.w * l.h);
                int loc = location % (l.w * l.h);
                return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) + entry * l.w * l.h + loc;
            }

            template<typename T>
            static void activate_array(T *x, const int n) {
                for (int i = 0; i < n; ++i, ++x) {
                    *x = T(1. / (1. + std::exp(neg(*x))));
                }
            }

            template<typename T>
            static void compute_run(layer l) {
                int b, n;
                auto data = reinterpret_cast<T *>(l.output);
                for (b = 0; b < l.batch; ++b) {
                    for (n = 0; n < l.n; ++n) {
                        int index = entry_index(l, b, n * l.w * l.h, 0);
                        activate_array<T>(data + index, 2 * l.w * l.h);
                        index = entry_index(l, b, n * l.w * l.h, 4);
                        activate_array<T>(data + index, (1 + l.classes) * l.w * l.h);
                    }
                }
            }

            int run(Stack &stack) override {
                std::vector<Tensor::Prototype> output;
                infer(stack, output);

                auto memory_device = MemoryDevice(CPU);

                auto x = stack[0].view(memory_device);
                auto &out = *stack.push(output[0], memory_device);

                // calculate out
                auto dest = out.weak_memory();
                auto src = x.weak_memory();
                memcpy(dest, src);

                layer l;
                l.h = x.size(2);
                l.w = x.size(3);
                l.outputs = out.size(1) * out.size(2) * out.size(3);
                l.classes = m_classes;
                l.batch = x.size(0);
                l.n = m_n;
                l.output = out.data();

                DTYPE dtype = out.dtype();
                switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { compute_run<TYPE>(l); break; }
                    DECLARE_COMPUTE_RUN(INT8, int8_t);
                    DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                    DECLARE_COMPUTE_RUN(INT16, int16_t);
                    DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                    DECLARE_COMPUTE_RUN(INT32, int32_t);
                    DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                    DECLARE_COMPUTE_RUN(INT64, int64_t);
                    DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                    DECLARE_COMPUTE_RUN(FLOAT32, float);
                    DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                    default: {
                        TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                        break;
                    }
                }
                out.pack({out, param.classes, param.mask, param.anchors});

                return 1;
            }

        private:
            int m_n;
            int m_classes;
            std::vector<int32_t> m_mask;
            std::vector<float> m_anchors;

            struct {
                Tensor classes;
                Tensor mask;
                Tensor anchors;
            } param;
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(Yolo, CPU, name::layer::yolo())
