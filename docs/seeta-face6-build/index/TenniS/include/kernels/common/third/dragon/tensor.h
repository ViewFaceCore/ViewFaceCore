//
// Created by kier on 2019/9/6.
//

#ifndef TENSORSTACK_THIRD_DRAGON_TENSOR_H
#define TENSORSTACK_THIRD_DRAGON_TENSOR_H

#include <core/tensor.h>

#include <numeric>

#include "type_meta.h"

namespace ts {
    namespace dragon {

		template <typename T, typename F>
		inline std::vector<T> cast_tensor(const std::vector<F> &vec) {
			std::vector<T> to_vec(vec.size());
			std::transform(vec.begin(), vec.end(), to_vec.begin(), [](const F &a) { return T(a); });
			return std::move(to_vec);
		}

        class TensorV0 {
        public:
            using self = TensorV0;

            TensorV0() = default;

            TensorV0(const ts::Tensor &tst) : m_tst(tst) {}

            operator ts::Tensor() const { return m_tst; }

            int64_t dim(int64_t i) const { return int64_t(m_tst.size(int(i))); }

            self *Reshape(DTYPE dtype, const std::vector<int64_t> &shape) {
				auto ts_shape = cast_tensor<int>(shape);
                auto count = std::accumulate(ts_shape.begin(), ts_shape.end(), 1, std::multiplies<int>());
                if (m_tst.dtype() == dtype && m_tst.count() == count) {
                    m_tst.reshape(ts_shape);
                    return this;
                }
                m_tst = ts::Tensor();
                m_tst = ts::Tensor(ts::Tensor::InFlow::DEVICE, dtype, ts_shape);
                return this;
            }

            template<typename T>
            self *Reshape(const std::vector<int64_t> &shape) {
                return Reshape(dtypeid<T>::id, shape);
            }

            template<typename T, typename Context>
            self *Reshape(const std::vector<int64_t> &shape) {
                return Reshape(dtypeid<T>::id, shape);
            }

            self *ReshapeLike(const self &other) {
                auto &shape = other.m_tst.sizes();
                return Reshape(other.m_tst.dtype(), std::vector<int64_t>(shape.begin(), shape.end()));
            }

            template<typename T>
            bool IsType() const { return m_tst.dtype() == dtypeid<T>::id; }

            int64_t count() const {
                return int64_t(m_tst.count());
            }

            template<typename T, typename Context>
            T *mutable_data() {
                if (!IsType<T>()) {
                    TS_LOG_ERROR << "Expected dtype = " << type_str(dtypeid<T>::id)
                                 << " got " << type_str(m_tst.dtype()) << eject;
                }
                return reinterpret_cast<T *>(this->mutable_data_ptr<Context>());
            }

            template<typename T, typename Context>
            const T *data() const {
                if (!IsType<T>()) {
                    TS_LOG_ERROR << "Expected dtype = " << type_str(dtypeid<T>::id)
                                 << " got " << type_str(m_tst.dtype()) << eject;
                }
                return reinterpret_cast<const T *>(this->const_data_ptr<Context>());
            }

            template<typename Context>
            void *mutable_data_ptr() {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::HOST);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::DEVICE);
                }
                m_tst.broadcast();
                return m_tst.data();
            }

            template<typename Context>
            const void *const_data_ptr() const {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::HOST);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    m_tst = m_tst.view(ts::Tensor::InFlow::DEVICE);
                }
                return m_tst.data();
            }

            std::string name() const { return ""; }

            ts::Tensor::Prototype meta() const { return m_tst.proto(); }

            friend class TensorObject;
        private:
            mutable ts::Tensor m_tst;

        };

        class TensorObject {
        public:
            mutable ts::Tensor m_tst;
            std::vector<int64_t> m_shape;
            mutable bool m_shape_updated = false;

            TensorObject() = default;

            TensorObject(const ts::Tensor &tensor)
                    : m_tst(tensor) {
                auto &shape = tensor.sizes();
                m_shape = std::vector<int64_t>(shape.begin(), shape.end());
            }

            DTYPE dtype() const { return m_tst.dtype(); }

            int64_t dim(int64_t i) const { return int64_t(m_tst.size(int(i))); }

            int64_t count() const { return int64_t(m_tst.count()); }

            bool updated() const { return m_shape_updated; }

            ts::Tensor t() const { return m_tst; }

            const std::vector<int64_t> &shape() const { return m_shape; }

            void reshape(const std::vector<int64_t> &shape) {
                m_shape_updated = m_shape.size() != shape.size();
                if (!m_shape_updated) {
                    for (size_t i = 0; i < shape.size(); ++i) {
                        if (m_shape[i] != shape[i]) {
                            m_shape_updated = true;
                            break;
                        }
                    }
                }
                m_shape = shape;
            }

            const void *view(const MemoryDevice &device) const {
                m_tst = m_tst.view(device);
                return m_tst.data();
            }

            void *mutable_view(const MemoryDevice &device) {
                m_tst = m_tst.view(device);
                m_tst.broadcast();
                return m_tst.data();
            }

            template<typename Context>
            const void *const_data_ptr() const {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    return view(MemoryDevice(CPU));
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    return view(ctx::of<Workbench>::ref().device().memory_device);
                }
                return m_tst.data();
            }

            template<typename Context>
            void *mutable_data_ptr() {
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    return mutable_view(MemoryDevice(CPU));
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    return mutable_view(ctx::of<Workbench>::ref().device().memory_device);
                }
                m_tst.broadcast();
                return m_tst.data();
            }

            void newdata(DTYPE dtype, const MemoryDevice &device, SyncMemoryController::shared flow) const {
                m_tst = ts::Tensor(flow, dtype, cast_tensor<int32_t>(m_shape), MemoryDevice(CPU));
            }

            void newdata(DTYPE dtype, const MemoryDevice &device) const {
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                newdata(dtype, device, flow);
            }

            void newdata(DTYPE dtype) const {
                newdata(dtype, MemoryDevice(CPU));
            }

            template<typename Context>
            void ensure_dtype_and_shape(DTYPE dtype) const {
                if (this->dtype() == dtype && !m_shape_updated) return;
                auto flow = ctx::of<Workbench>::ref().runtime().flow();
                if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                    newdata(dtype, MemoryDevice(CPU), flow);
                } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                    auto device = ctx::of<Workbench>::ref().device().memory_device;
                    newdata(dtype, device, flow);
                }
                m_shape_updated = false;
            }

            template<typename T, typename Context>
            void ensure_shape() const {
                ensure_dtype_and_shape<Context>(dtypeid<T>::id);
            }

            template<typename T>
            bool IsType() const { return m_tst.dtype() == dtypeid<T>::id; }

            template<typename T, typename Context>
            T *mutable_data() {
                ensure_shape<T, Context>();
                auto data = mutable_data_ptr<Context>();
                return reinterpret_cast<T *>(data);
            }

            template<typename T, typename Context>
            const T *data() const {
                if (!IsType<T>()) {
                    TS_LOG_ERROR << "Expected dtype = " << type_str(dtypeid<T>::id)
                                 << " got " << type_str(m_tst.dtype()) << eject;
                }
                ensure_shape<T, Context>();
                auto data = const_data_ptr<Context>();
                return reinterpret_cast<const T *>(data);
            }
        };

        class Tensor {
        public:
            using self = Tensor;

            Tensor() : m_obj(std::make_shared<TensorObject>()) {};

            Tensor(const ts::Tensor &tst) : m_obj(std::make_shared<TensorObject>(tst)) {}

            operator ts::Tensor() const { return m_obj->t(); }

            int64_t dim(int64_t i) const { return m_obj->dim(i); }

            self *Reshape(const std::vector<int64_t> &shape) {
                m_obj->reshape(shape);
                return this;
            }

            self *ReshapeLike(const self &other) {
                m_obj->reshape(other.m_obj->shape());
                return this;
            }

            template<typename T>
            bool IsType() const { return m_obj->IsType<T>(); }

            int64_t count() const {
                return m_obj->count();
            }

            template<typename T, typename Context>
            T *mutable_data() {
                return m_obj->mutable_data<T, Context>();
            }

            template<typename T, typename Context>
            const T *data() const {
                return m_obj->data<T, Context>();
            }

            std::string name() const { return ""; }

            ts::Tensor::Prototype meta() const { return m_obj->t().proto(); }

            void dispose() {
                m_obj = std::make_shared<TensorObject>();
            }

        private:
            std::shared_ptr<TensorObject> m_obj;
        };
    }
}

#define XIsType(x, T) ((x).template IsType<T>())

#endif //TENSORSTACK_THIRD_DRAGON_TENSOR_H
