//
// Created by kier on 2020/6/2.
//

#ifndef TENNIS_BLOB_HPP
#define TENNIS_BLOB_HPP

#include "core/tensor.h"

namespace ts {
    namespace caffe {

        template <typename T>
        class Blob {
        private:
            using Dtype = T;
            using self = Blob;

            Tensor data_cpu;
        public:
            int count() const {
                return data_cpu.count();
            }

            const Dtype *cpu_data() const {
                return data_cpu.data<Dtype>();
            }
            Dtype *mutable_cpu_data() {
                return data_cpu.data<Dtype>();
            }

            void Reshape(const std::vector<int> &shape) {
                Tensor::Prototype proto(dtypeid<Dtype>::id, shape);
                if (data_cpu.count() == proto.count()) {
                    data_cpu = data_cpu.reshape(shape);
                } else {
                    data_cpu = Tensor(Tensor::InFlow::HOST, proto);
                }
            }

            const std::vector<int> shape() const {
                return data_cpu.sizes().std();
            }

            void dispose() {
                data_cpu = Tensor();
            }

            const self *operator->() const { return this; }

            self *operator->() { return this; }

            Tensor tensor() const { return data_cpu; }
        };

    }
}

#endif //TENNIS_BLOB_HPP
