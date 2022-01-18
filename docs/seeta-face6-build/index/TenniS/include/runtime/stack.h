//
// Created by kier on 2018/5/25.
//

#ifndef TENSORSTACK_RUNTIME_STACK_H
#define TENSORSTACK_RUNTIME_STACK_H

#include "core/device.h"
#include "core/sync/sync_controller.h"
#include "core/dtype.h"
#include "core/tensor.h"
#include "global/hard_converter.h"
#include "utils/assert.h"

#include <vector>
#include <deque>
#include <stack>


namespace ts {

    class TS_DEBUG_API Stack {
    public:
        using self = Stack;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

        /**
         * build stack on device, using dynamic memory controller
         * @param device memory device
         * @note default need_lock is false
         */
        explicit Stack(const MemoryDevice &device);

        /**
         * build stack on device, using dynamic memory controller
         * @param device memory device
         * @param need_lock if need lock to control memory share with this stack
         */
        explicit Stack(const MemoryDevice &device, bool need_lock);

        /**
         * build stack on device, using given memory controller
         * @param device memory device
         * @param controller given memory controller
         */
        explicit Stack(const MemoryDevice &device, const SyncMemoryController::shared &controller);

        /**
         * Build tensor with dtype and shape
         * @param dtype new tensor's dtype
         * @param shape new tensor's dtype
         * @return new tensor
         */
        Tensor make(DTYPE dtype, const Shape &shape);

        /**
         * Build tensor with dtype and shape on device
         * @param dtype new tensor's dtype
         * @param shape new tensor's dtype
         * @param device new tensor's device
         * @return new tensor
         */
        Tensor make(DTYPE dtype, const Shape &shape, const MemoryDevice &device);

        /**
         * Build tensor with proto
         * @param proto new tensor's proto
         * @return new tensor
         */
        Tensor make(const Tensor::Prototype &proto) {
            return this->make(proto.dtype(), proto.sizes());
        }

        /**
         * Build tensor with proto
         * @param proto new tensor's proto
         * @param device new tensor's device
         * @return new tensor
         */
        Tensor make(const Tensor::Prototype &proto, const MemoryDevice &device) {
            return this->make(proto.dtype(), proto.sizes(), device);
        }

        /**
         * Build tensor with proto
         * @param proto new tensor's proto
         * @return new tensor
         */
        Tensor make(const TensorPrototype &proto);

        /**
         * Build tensor with proto
         * @param proto new tensor's proto
         * @param device new tensor's device
         * @return new tensor
         */
        Tensor make(const TensorPrototype &proto, const MemoryDevice &device);

        /**
         *
         * @param in_flow
         * @param proto
         * @return new Tensor
         * Notice: moptional have context:DeviceContext
         * DeviceContext must given with in_flow=InFlow::DEVICE,
         * if in_flow=InFlow::HOST, use CPU:0 memory device.
         */
        Tensor make(Tensor::InFlow in_flow, const Tensor::Prototype &proto);

        /**
         *
         * @param in_flow
         * @param proto
         * @return new Tensor
         * Notice: moptional have context:DeviceContext
         * DeviceContext must given with in_flow=InFlow::DEVICE,
         * if in_flow=InFlow::HOST, use CPU:0 memory device.
         */
        Tensor make(Tensor::InFlow in_flow, const TensorPrototype &proto);

        /**
         * Push tensor with dtype and shape
         * @param dtype new tensor's dtype
         * @param shape new tensor's shape
         * @return pointer to new tensor
         */
        Tensor *push(DTYPE dtype, const Shape &shape) {
            return this->push(this->make(dtype, shape));
        }

        /**
         * Push tensor with dtype and shape on device
         * @param dtype new tensor's dtype
         * @param shape new tensor's shape
         * @param shape new tensor's device
         * @return pointer to new tensor
         */
        Tensor *push(DTYPE dtype, const Shape &shape, const MemoryDevice &device) {
            return this->push(this->make(dtype, shape, device));
        }

        /**
         * Push tensor with proto
         * @param proto new tensor's proto
         * @return pointer to new tensor
         */
        Tensor *push(const Tensor::Prototype &proto) {
            return this->push(proto.dtype(), proto.sizes());
        }

        /**
         * Push tensor with proto
         * @param proto new tensor's proto
         * @param device new tensor's device
         * @return pointer to new tensor
         */
        Tensor *push(const Tensor::Prototype &proto, const MemoryDevice &device) {
            return this->push(proto.dtype(), proto.sizes(), device);
        }

        /**
         * Push tensor with proto
         * @param proto new tensor's proto
         * @return pointer to new tensor
         */
        Tensor *push(const TensorPrototype &proto) {
            return this->push(this->make(proto));
        }

        /**
         * Push tensor with proto
         * @param proto new tensor's proto
         * @param device new tensor's device
         * @return pointer to new tensor
         */
        Tensor *push(const TensorPrototype &proto, const MemoryDevice &device) {
            return this->push(this->make(proto, device));
        }

        /**
         * Push given tensor to stack
         * @param tensor given tensor
         * @return pointer to given tensor
         */
        Tensor *push(const Tensor &tensor);

        /**
         * Push stack[i] to the top of stack
         * @param i index of need push tensor
         * @return pointer to index tensor
         */
        Tensor *push(int i) { return this->push(*this->index(i)); }

        /**
         * clone_push means push an clone of tensor
         * @param tensor param to clone
         * @return return cloned tensor
         */
        Tensor *clone_push(const Tensor &tensor);

        /**
         * clone_push means push an clone of tensor, on memory device
         * @param tensor param to clone
         * @param device memory device
         * @return return cloned tensor
         */
        Tensor *clone_push(const Tensor &tensor, const MemoryDevice &device);

        /**
         * clone the stack[i] on the top
         * @param i index of clone tensor
         * @return pointer to new tensor
         */
        Tensor *clone(int i) { return this->clone_push(*this->index(i)); }

        /**
         * clone the stack[i] on the top, on memory device
         * @param i index of clone tensor
         * @param device memory device
         * @return pointer to new tensor
         */
        Tensor *clone(int i, const MemoryDevice &device) { return this->clone_push(*this->index(i), device); }

        /**
         * get stack[i]
         * @param i index to tensor, can be negative
         * @return stack[i]
         * @note if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
         */
        Tensor *index(int i);

        /**
         * get stack[i]
         * @param i index to tensor, can be negative
         * @return stack[i]
         * @note if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
         */
        const Tensor *index(int i) const;

        /**
         * get stack[i]
         * @param i index to tensor, can be negative
         * @return stack[i]
         * @note if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
         */
        Tensor &operator[](int i) { return *index(i); }

        /**
         * get stack[i]
         * @param i index to tensor, can be negative
         * @return stack[i]
         * @note if i >= 0, then i is bottom_up_index; else if i < 0, the i is top_down_index
         */
        const Tensor &operator[](int i) const { return *index(i); }

        /**
         * get stack[i]
         * @param i index to tensor
         * @return stack[i]
         */
        Tensor &operator[](size_t i) { return *index(int(i)); }

        /**
         * get stack[i]
         * @param i index to tensor
         * @return stack[i]
         */
        const Tensor &operator[](size_t i) const { return *index(int(i)); }

        /**
         * get stack[-1]
         * @return stack[-1]
         */
        Tensor *top();

        /**
         * get stack[-1]
         * @return stack[-1]
         */
        const Tensor *top() const;

        /**
         * erase pop_size tensor top of stack
         * @param pop_size pop size
         */
        void pop(size_t pop_size);

        /**
         * erase top of stack
         */
        void pop();

        /**
         * get size of stack, on the base
         * @return
         */
        size_t size() const;

        /**
         * erase stack[i]
         * @param i index of tensor ready to remove
         */
        void erase(int i);

        /**
         * remove [beg, end)
         * @param beg begin of the remove range
         * @param end end of the remove range
         */
        void erase(int beg, int end);

        /**
         * clear all items above base
         */
        void clear() { this->erase(0, static_cast<int>(this->size())); }

        /**
         * get now running base, mean the bottom of stack
         * @return running base
         */
        size_t base() const;

        /**
         * change base
         * @param new_base new base
         * @return old base
         */
        size_t rebase(size_t new_base);

        /**
         * change base, then call the pop_base can resume older base
         * @param i new base on stack[i]
         */
        void push_base(int i);

        /**
         * resume base before the last call of push_base
         */
        void pop_base();

        /**
         * get converter on stack's memory
         * @return memory converter
         */
        HardConverter::function converter() const;

        std::deque<Tensor>::const_iterator begin() const {
            return m_stack.begin() + m_base;
        }

        std::deque<Tensor>::iterator begin() {
            return m_stack.begin() + m_base;
        }

        std::deque<Tensor>::const_iterator end() const {
            return m_stack.end();
        }

        std::deque<Tensor>::iterator end() {
            return m_stack.end();
        }

    private:
        // size_t relative2absolute(int i) const;

        Device m_device;                          ///< running tensor device, compute on it
        SyncMemoryController::shared m_controller;    ///< tensor memory backend
        std::deque<Tensor> m_stack;               ///< saving all tensor
        size_t m_base = 0;                        ///< the running control base
        std::stack<size_t> m_base_stack;          ///< save each call base

        mutable HardConverter::function m_converter = nullptr;    ///< convert memory in stack
    };
}


#endif //TENSORSTACK_RUNTIME_STACK_H
