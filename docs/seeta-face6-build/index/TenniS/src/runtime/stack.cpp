//
// Created by kier on 2018/5/25.
//

#include "runtime/stack.h"
#include "core/device_context.h"

namespace ts {

#define relative2absolute(i) ((i) >= 0 ? m_base + (i) : m_stack.size() + (i))

    Stack::Stack(const MemoryDevice &device)
            : Stack(device, DynamicSyncMemoryController::Make(device)) {}

    Stack::Stack(const MemoryDevice &device, bool need_lock)
            : Stack(device, DynamicSyncMemoryController::Make(device, need_lock)) {}

    Stack::Stack(const MemoryDevice &device, const SyncMemoryController::shared &controller)
            : m_device(device), m_controller(controller) {}

    Tensor Stack::make(DTYPE dtype, const Shape &shape) {
        return Tensor(m_controller, dtype, shape);
    }

    Tensor Stack::make(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
        return Tensor(m_controller, dtype, shape, device);
    }

    Tensor Stack::make(const TensorPrototype &proto) {
        Tensor packed;
        auto count = proto.fields_count();
        packed.refield(count);
        for (decltype(count) i = 0; i < count; ++i) {
            packed.field(i, this->make(proto.field(i)));
        }
        return std::move(packed);
    }

    Tensor Stack::make(const TensorPrototype &proto, const MemoryDevice &device) {
        Tensor packed;
        auto count = proto.fields_count();
        packed.refield(count);
        for (decltype(count) i = 0; i < count; ++i) {
            packed.field(i, this->make(proto.field(i), device));
        }
        return std::move(packed);
    }

    Tensor *Stack::push(const Tensor &tensor) {
        // removed this check, supporting cross device computing
        // TS_AUTO_CHECK(tensor.device() == this->m_device);
        this->m_stack.push_back(tensor);
        return &this->m_stack.back();
    }

    Tensor *Stack::clone_push(const Tensor &tensor) {
        return this->push(tensor.clone(this->m_controller));
    }

    Tensor *Stack::clone_push(const Tensor &tensor, const MemoryDevice &device) {
        return this->push(tensor.clone(this->m_controller, device));
    }

    Tensor *Stack::index(int i) { return &this->m_stack.at(relative2absolute(i)); }

    const Tensor *Stack::index(int i) const { return &this->m_stack.at(relative2absolute(i)); }

    Tensor *Stack::top() { return &this->m_stack.back(); }

    const Tensor *Stack::top() const { return &this->m_stack.back(); }

    void Stack::pop(size_t pop_size) {
        auto it_end = this->m_stack.end();
        auto it_beg = it_end - pop_size;
        this->m_stack.erase(it_beg, it_end);
    }

    void Stack::pop() { this->m_stack.pop_back(); }

    size_t Stack::size() const { return m_stack.size() - m_base; }

    void Stack::erase(int i) {
        auto it = this->m_stack.begin() + relative2absolute(i);
        this->m_stack.erase(it);
    }

    void Stack::erase(int beg, int end) {
        auto beg_it = this->m_stack.begin() + relative2absolute(beg);
        auto end_it = this->m_stack.begin() + relative2absolute(end);
        // TS_AUTO_CHECK(end_it - beg_it >= 0);
        this->m_stack.erase(beg_it, end_it);
    }

    size_t Stack::base() const { return m_base; }

    size_t Stack::rebase(size_t new_base) {
        std::swap(m_base, new_base);
        return new_base;
    }

    void Stack::push_base(int i) {
        this->m_base_stack.push(this->rebase(relative2absolute(i)));
    }

    void Stack::pop_base() {
        if (this->m_base_stack.empty()){
            this->rebase(0);
        } else {
            this->rebase(this->m_base_stack.top());
            this->m_base_stack.pop();
        }
    }

    HardConverter::function Stack::converter() const {
        if (this->m_converter == nullptr) {
            this->m_converter = HardConverter::Query(m_device.type(), m_device.type());
            TS_AUTO_CHECK(this->m_converter != nullptr);
        }
        return this->m_converter;
    }

    Tensor Stack::make(Tensor::InFlow in_flow, const Tensor::Prototype &proto) {
        switch (in_flow) {
            case Tensor::InFlow::HOST: {
                return make(proto, MemoryDevice(CPU, 0));
                break;
            }
            case Tensor::InFlow::DEVICE: {
                auto device = ctx::of<DeviceContext>::ref().memory_device;
                return make(proto, device);
                break;
            }
        }
        return make(proto);
    }

    Tensor Stack::make(Tensor::InFlow in_flow, const TensorPrototype &proto) {
        switch (in_flow) {
            case Tensor::InFlow::HOST: {
                return make(proto, MemoryDevice(CPU, 0));
                break;
            }
            case Tensor::InFlow::DEVICE: {
                auto device = ctx::of<DeviceContext>::ref().memory_device;
                return make(proto, device);
                break;
            }
        }
        return make(proto);
    }
}
