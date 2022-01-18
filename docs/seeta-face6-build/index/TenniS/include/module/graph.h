//
// Created by kier on 2018/10/15.
//

#ifndef TENSORSTACK_MODULE_GRAPH_H
#define TENSORSTACK_MODULE_GRAPH_H

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "utils/except.h"
#include "utils/assert.h"

#include "bubble.h"

#include "utils/ctxmgr_lite.h"

namespace ts {
    template <typename T>
    class TS_DEBUG_API LinkedValue {
    public:
        using self = LinkedValue;
        using shared = std::shared_ptr<self>;
        using weak = std::weak_ptr<self>;

        template<typename... Args>
        explicit LinkedValue(Args &&...args)
                : m_value(std::forward<Args>(args)...) {}

        virtual ~LinkedValue() = default;

        const std::vector<weak> &inputs() const { return m_inputs; }

        const std::vector<weak> &outputs() const { return m_outputs; }

        static void Link(const weak &node, const std::vector<weak> &inputs) {
            auto output = node.lock();
            if (!output) throw NullPointerException("Link expired node");
            output->m_inputs.resize(inputs.size());
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i].lock();
                if (!input) throw NullPointerException("Link expired node");
                input->m_outputs.push_back(output);
                output->m_inputs[i] = input;
            }
        }

        static void ReplaceOutput(const weak &old_node, const weak &new_node) {
            auto old_ptr = old_node.lock();
            if (!old_ptr) throw NullPointerException("Link expired node");
            auto new_ptr = new_node.lock();
            if (!new_ptr) throw NullPointerException("Link expired node");
            for (auto &output : old_ptr->outputs()) {
                auto output_ptr = output.lock();
                for (auto &output_s_input : output_ptr->m_inputs) {
                    if (output_s_input.lock() != old_ptr) continue;
                    output_s_input = new_node;
                    break;
                }
                new_ptr->m_outputs.push_back(output);
            }
            old_ptr->m_outputs.clear();
        }

        T &value() { return m_value; }

        const T &value() const { return m_value; }

        std::string str() const {
            std::ostringstream oss;
            oss << "<Node: " << this->value() << ">";
            return oss.str();
        }

        std::string repr() const { return this->str(); }

        T *ptr() { return &m_value; }

        const T *ptr() const { return &m_value; }

        T &ref() { return m_value; }

        const T &ref() const { return m_value; }

    private:
        T m_value;
        std::vector<weak> m_inputs;
        std::vector<weak> m_outputs;
    };

    using LinkedBubble = LinkedValue<Bubble>;

    /**
     * Node only support single output
     * Use Pack node support multi output, like:
     *     c = func1(a, b) # c is pack node
     *     c:1 = unpack(c, 1)   # get c's 1st output
     *     c:2 = unpack(c, 2)   # get c's 2nd output
     *  Notice: The c is pack(c:1, c:2) node, and the unpack method's first parameter must be pack node
     *  TODO: supporting edit graph, not just a link
     */
    class TS_DEBUG_API Node {
    public:
        using self = Node;

        friend class Graph;

        Node(const self &) = default;

        Node &operator=(const self &) = default;

        std::vector<Node> inputs() const {
            auto ptr = m_ptr.lock();
            if (!ptr) throw NullPointerException("Getting expired node's inputs");
            auto raw_vector = ptr->inputs();
            std::vector<Node> out_vector;
            out_vector.reserve(raw_vector.size());
            for (auto &node : raw_vector) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

        std::vector<Node> outputs() const {
            auto ptr = m_ptr.lock();
            if (!ptr) throw NullPointerException("Getting expired node's outputs");
            auto raw_vector = ptr->outputs();
            std::vector<Node> out_vector;
            out_vector.reserve(raw_vector.size());
            for (auto &node : raw_vector) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

        Node input(int i) const { return inputs()[i]; }

        Node input(size_t i) const { return inputs()[i]; }

        void *ptr() const { return m_ptr.lock().get(); }

//        template<typename T>
//        T *ptr();
//
//        template<typename T>
//        const T *ptr() const { return const_cast<self *>(this)->ptr<T>(); }

//        template<typename T>
//        T &ref() {
//            auto value_ptr = this->ptr<T>();
//            if (value_ptr == nullptr) throw NullPointerException("Getting reference from null pointer");
//            return *value_ptr;
//        }
//
//        template<typename T>
//        const T &ref() const { return const_cast<self *>(this)->ref<T>(); }

        /**
         * Link node perform as (inputs[0], ..., inputs[n - 1]) -> node
         * @param node output node
         * @param inputs input nodes
         */
        static void Link(const Node &node, const std::vector<Node> &inputs) {
            std::vector<LinkedBubble::weak> raw_inputs;
            raw_inputs.reserve(inputs.size());
            for (auto &input : inputs) raw_inputs.emplace_back(LinkedBubble::weak(input));
            LinkedBubble::Link(node.m_ptr, raw_inputs);
        }

        /**
         * Replace all output nodes linked with old_node to new_node
         * @param old_node the node ready to be replaced
         * @param new_node the new node
         */
        static void ReplaceOutput(const Node &old_node, const Node &new_node) {
            LinkedBubble::ReplaceOutput(old_node.m_ptr, new_node.m_ptr);
        }

        std::string str() const {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) return "<Node: nil>";
            return raw_ptr->str();
        }

        std::string repr() const {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) return "<Node: nil>";
            return raw_ptr->repr();
        }

        Bubble &bubble() {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) throw NullPointerException("Getting expired node's bubble");
            return raw_ptr->value();
        }

        const Bubble &bubble() const {
            auto raw_ptr = m_ptr.lock();
            if (!raw_ptr) throw NullPointerException("Getting expired node's bubble");
            return raw_ptr->value();
        }

        Bubble *operator->() {
            return &bubble();
        }

        const Bubble *operator->() const {
            return &bubble();
        }

        Bubble &operator*() {
            return bubble();
        }

        const Bubble &operator*() const {
            return bubble();
        }

    private:
        explicit Node(LinkedBubble::weak ptr) : m_ptr(std::move(ptr)) {}

        explicit operator LinkedBubble::weak() const { return m_ptr; }

        LinkedBubble::weak m_ptr;
    };

//    template<typename T>
//    T *Node::ptr() {
//        TS_LOG_ERROR << "Using not recommended API, please use \"Node::bubble\" instead.";
//        auto raw_ptr = m_ptr.lock();
//        if (!raw_ptr) return nullptr;
//        return raw_ptr->ptr();
//    }

    inline std::ostream &operator<<(std::ostream &out, const Node &node) {
        return out << node.str();
    }

    /**
     * Graph, only saving nodes,
     * The Node generated by Graph will be disabled after destruction
     */
    class TS_DEBUG_API Graph : public SetupContext<Graph> {
    public:
        using self = Graph;    ///< self class
        using shared = std::shared_ptr<self>;  ///< smart pointer

//        template<typename T, typename... Args>
//        Node make(Args &&...args) {
//            TS_LOG_ERROR << "Using not recommended API, please use \"Graph::make\" instead.";
//            auto node = std::make_shared<LinkedBubble>(std::forward<Args>(args)...);
//            m_nodes.push_back(node);
//            return Node(node);
//        }

        template<typename... Args>
        Node make(Args &&...args) {
            auto node = std::make_shared<LinkedBubble>(std::forward<Args>(args)...);
            m_nodes.push_back(node);
            return Node(node);
        }

        std::vector<Node> nodes() const {
            std::vector<Node> out_vector;
            out_vector.reserve(m_nodes.size());
            for (auto &node : m_nodes) out_vector.emplace_back(Node(node));
            return std::move(out_vector);
        }

    private:
        std::vector<LinkedBubble::shared> m_nodes;
    };

    inline bool operator==(const Node &lhs, const Node &rhs) { return lhs.ptr() == rhs.ptr(); }

    inline bool operator!=(const Node &lhs, const Node &rhs) { return lhs.ptr() != rhs.ptr(); }

    inline bool operator<(const Node &lhs, const Node &rhs) { return lhs.ptr() < rhs.ptr(); }

    inline bool operator>(const Node &lhs, const Node &rhs) { return lhs.ptr() > rhs.ptr(); }

    inline bool operator<=(const Node &lhs, const Node &rhs) { return lhs.ptr() <= rhs.ptr(); }

    inline bool operator>=(const Node &lhs, const Node &rhs) { return lhs.ptr() >= rhs.ptr(); }

    TS_DEBUG_API std::ostream &plot_graph(std::ostream &stream, const std::vector<Node> &nodes);
}

namespace std {
    template<>
    struct hash<ts::Node> {
        std::size_t operator()(const ts::Node &key) const {
            using std::size_t;
            using std::hash;

            return hash<void *>()(key.ptr());
        }
    };
}

#endif //TENSORSTACK_MODULE_GRAPH_H
