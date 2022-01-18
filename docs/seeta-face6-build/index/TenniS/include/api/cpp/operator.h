//
// Created by kier on 19-5-8.
//

#ifndef TENNIS_API_CPP_OPERATOR_H
#define TENNIS_API_CPP_OPERATOR_H

#include "../operator.h"
#include "tensor.h"

namespace ts {
    namespace api {
        template<typename T>
        class NewAndFree {
        public:
            static void *New() { return new T; }

            static void Free(const void *obj) { delete reinterpret_cast<const T *>(obj); }
        };

        class OperatorParams {
        public:
            explicit OperatorParams(const ts_OperatorParams *params)
                    : m_params(params) {}

            Tensor get(const char *param) const {
                return Tensor::NewRef(ts_OperatorParams_get(m_params, param));
            }

            Tensor get(const std::string &param) const { return get(param.c_str()); }

        private:
            const ts_OperatorParams *m_params;
        };

        inline std::vector<Tensor> borrowed_tensors(int32_t argc, ts_Tensor **argv) {
            std::vector<Tensor> args;
            for (int i = 0; i < argc; ++i) {
                args.emplace_back(Tensor::BorrowedRef(argv[i]));
            }
            return std::move(args);
        }

        class Operator {
        public:
            using self = Operator;

            virtual ~Operator() = default;

            virtual void init(const OperatorParams &params, ts_OperatorContext *context) = 0;

            virtual std::vector<std::pair<DTYPE, Shape>> infer(const std::vector<Tensor> &args, ts_OperatorContext *context) = 0;

            virtual std::vector<Tensor> run(const std::vector<Tensor> &args, ts_OperatorContext *context) = 0;

            static void Throw(const std::string &message) {
                ts_Operator_Throw(message.c_str());
            }

            static void Throw(const std::string &message, const std::string &filename, int32_t line_number) {
                ts_Operator_ThrowV2(message.c_str(), filename.c_str(), line_number);
            }

            static void Init(void *op, const ts_OperatorParams *params, ts_OperatorContext *context) {
                auto obj = reinterpret_cast<Operator *>(op);
                obj->init(OperatorParams(params), context);
            }

            static ts_Tensor *Infer(void *op, int32_t argc, ts_Tensor **argv, ts_OperatorContext *context) {
                auto obj = reinterpret_cast<Operator *>(op);
                auto proto = obj->infer(borrowed_tensors(argc, argv), context);
                if (proto.empty()) return nullptr;
                std::vector<int32_t> data;
                data.emplace_back(int32_t(proto.size()));
                for (auto &pair : proto) {
                    data.emplace_back(int32_t(pair.first));
                    data.emplace_back(int32_t(pair.second.size()));
                    for (auto dim : pair.second) {
                        data.emplace_back(dim);
                    }
                }
                auto count = int32_t(data.size());
                auto packed_tensor = ts_new_Tensor(&count, 1, TS_INT32, data.data());
                // TS_API_AUTO_CHECK(packed_tensor != nullptr); // check in tennis side
                return packed_tensor;
            }

            static ts_Tensor *Run(void *op, int32_t argc, ts_Tensor **argv, ts_OperatorContext *context) {
                auto obj = reinterpret_cast<Operator *>(op);
                auto fields = obj->run(borrowed_tensors(argc, argv), context);
                if (fields.empty()) return nullptr;
                std::vector<ts_Tensor *> cfields;
                for (auto &field : fields) {
                    cfields.emplace_back(field.get_raw());
                }
                auto packed_raw = ts_Tensor_pack(cfields.data(), int32_t(cfields.size()));
                // TS_API_AUTO_CHECK(packed_raw != nullptr);    // check in tennis side
                return packed_raw;
            }
        };

        /**
         *
         * @tparam T must be the subclass of Operator
         * @param device register device
         * @param op register op
         */
        template<typename T>
        inline void RegisterOperator(const std::string &device, const std::string &op) {
            ts_Operator_Register(device.c_str(), op.c_str(),
                                 &NewAndFree<T>::New, &NewAndFree<T>::Free,
                                 &T::Init, &T::Infer, &T::Run);
        }
    }
}

#define TS_THROW(message) \
    ts::api::Operator::Throw((message), __FILE__, __LINE__)

#endif //TENSORSTACK_OPERATOR_H
