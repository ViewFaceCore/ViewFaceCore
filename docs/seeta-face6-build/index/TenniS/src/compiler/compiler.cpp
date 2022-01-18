//
// Created by kier on 2018/10/17.
//

#include "compiler/compiler.h"
#include "module/module.h"
#include "runtime/instruction.h"

#include <deque>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <algorithm>
#include <backend/name.h>

#include "runtime/instruction/instruction_factory.h"
#include "runtime/instruction/stack_instruction.h"
#include "runtime/instruction/tensor_instruction.h"
#include "global/operator_factory.h"
#include "global/memory_device.h"
#include "core/tensor_builder.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"

#include "module/menu.h"
#include "frontend/intime.h"
#include "compiler/zipper.h"
#include "compiler/translater.h"



namespace ts {
    template <typename K, typename V>
    using map = std::unordered_map<K, V>;
    template <typename K>
    using set = std::unordered_set<K>;


    Compiler::Compiler(const ComputingDevice &computing_device)
            : m_computing_device(computing_device) {
    }

    // TODO: speed this function up
    static map<Node, set<Node>> build_node_refs(const std::vector<Node> &nodes) {
        auto computation_schedule = Module::list_reference_nodes(nodes);

        // build refs
        map<Node, set<Node>> map_node_refs;
        for (auto &node : computation_schedule) {
            set<Node> refs;
            for (auto &input : node.first.inputs()) {
                refs.insert(input);
                auto &input_refs = map_node_refs[input];
                refs.insert(input_refs.begin(), input_refs.end());
            }
            map_node_refs.insert(std::make_pair(node.first, std::move(refs)));
        }

        return map_node_refs;
    }

    std::vector<Instruction::shared> Compiler::convert_operator_instruction(const Node &node) {
        auto &bubble = node.bubble();

        // step 1: check inner InstructionCreator
        auto icreator = InstructionCreator::Query(bubble.op());
        if (icreator != nullptr) {
            return icreator(node);
        }
        auto creator = OperatorCreator::Query(m_computing_device.type(), bubble.op(), false);

        if (creator == nullptr) TS_LOG_ERROR << "Not supported operator " << bubble.op() << eject;
        std::string description = bubble.op() + "(in=" + std::to_string(node.inputs().size()) + ", out=" +
                                  std::to_string(1) + ")";
        auto op = creator();

        for (auto &param : bubble.params()) {
            op->set(param.first, param.second);
        }
        try {
            op->init();
        } catch (const Exception &e) {
            TS_LOG_ERROR << "While initializing " << bubble.op() << ":" << bubble.name() << " got Exception: " << e.what() << eject;
        }
        std::vector<Instruction::shared> instructions;
        auto op_inst = std::make_shared<OperatorInstruction>(op, int(node.inputs().size()), int(bubble.output_count()), description);
        op_inst->bind_creator(creator);
        instructions.emplace_back(std::move(op_inst));
        if (bubble.output_count() != 1) {
            TS_LOG_ERROR << "All operators' output count must be 1." << eject;
        }
        return std::move(instructions);
    }

    // TODO: inputs only support Parameter, try support other op
    InstructionBlock Compiler::compile(const std::vector<Node> &raw_inputs, const std::vector<Node> &raw_outputs,
            const std::string &options) {
        DeviceContext device_context(m_computing_device);
        ctx::bind<DeviceContext> _bind_device_context(device_context);

        auto inputs = raw_inputs;
        auto outputs = raw_outputs;

        // std::cout << "+++++++++++++++++ original graph ++++++++++++++++++++++" << std::endl;
        // plot_graph(std::cout, outputs);
        Graph temp_graph;
        ctx::bind<Graph> _bind_graph(temp_graph);
        
        // zip graph
        {
            Zipper zipper(m_computing_device, options);
            outputs = zipper.zip(outputs);
        }

        // const graph
        {
            std::vector<Node> const_outputs;
            run_const_nodes(outputs, const_outputs);
            outputs = const_outputs;
        }

        // std::cout << "+++++++++++++++++ const graph ++++++++++++++++++++++" << std::endl;
        // plot_graph(std::cout, outputs);

        // std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

        InstructionBlock block;
        block.nargs = int(inputs.size());
        block.nresults = int(outputs.size());

        // transform graph, make other graph from different framework to TS framework

        // compile graph, check node to computing device support nodes
        // 考虑处理inplace操作符，加入copy节点，保证inplace操作不会对其他节点造成影响

        // build refs
        // save if compile a node, whose nodes needed directly or indirectly
        map<Node, set<Node>> map_node_refs = build_node_refs(outputs);

        map<Node, int> map_node_data_sagment_index;

        // convert graph to instructions
        std::deque<Node> simulator;
        map<Node, size_t> working_nodes;
        size_t unsolved_node_count = 0;

        /**
         * \brief add node data sagment
         * \return data index
         */
        auto push_data_sagment = [&](Node node) -> int {
            auto node_it = map_node_data_sagment_index.find(node);
            if (node_it != map_node_data_sagment_index.end()) {
                return node_it->second;
            }
            auto &bubble = node.bubble();
            auto value = bubble.get(name::value);
            auto data_index = int(block.data_segment.size());
            if (bubble.has(name::device)) {
                block.data_segment.push_back(DeviceTensor(value, tensor::to_string(bubble.get(name::device))));
            } else {
                block.data_segment.push_back(DeviceTensor(value));
            }
            map_node_data_sagment_index.insert(std::make_pair(node, data_index));
            return data_index;
        };

        /**
         * \brief add node to simulator
         * \param node node ready to push
         */
        auto simulator_push = [&](Node node) {
            auto &bubble = node.bubble();
            size_t i = simulator.size();
            auto it = working_nodes.find(node);
            if (it == working_nodes.end()) {
                working_nodes.insert(std::make_pair(node, i));
            } else {
                if (i < it->second) it->second = i;
            }
            if (bubble.op() != Bubble::Parameter) {
                ++unsolved_node_count;
            }
            simulator.push_back(node);
        };

        /**
         * \brief pop node from simulator
         */
        auto simulator_pop = [&]() {
            if (simulator.empty()) return;
            auto node = simulator.back();
            auto &bubble = node.bubble();
            size_t i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it != working_nodes.end() && it->second == i) {
                working_nodes.erase(node);
            }
            if (bubble.op() != Bubble::Parameter) {
                --unsolved_node_count;
            }
            simulator.pop_back();
        };

        /**
         * \brief swap nodes at i and j
         * \param i
         * \param j
         * \param only used in swap last in node and last not in node
         */
        auto simulator_swap = [&](int i, int j) {
            // pop front
            auto index_i = i >= 0 ? size_t(i) : size_t(int64_t(simulator.size()) + i);
            auto index_j = j >= 0 ? size_t(j) : size_t(int64_t(simulator.size()) + j);

            auto nodei = simulator[index_i];
            auto nodej = simulator[index_j];

            auto nodei_it = working_nodes.find(nodei);
            if (nodei_it != working_nodes.end() && nodei_it->second == index_i) {
                nodei_it->second = index_j;
            }
            auto nodej_it = working_nodes.find(nodej);
            if (nodej_it != working_nodes.end() && nodej_it->second == index_j) {
                nodej_it->second = index_i;
            }

            simulator[index_i] = nodej;
            simulator[index_j] = nodei;
        };

        /**
         * \brief find last unsolved node index
         * \return found index
         * \note return -1 if failed
         */
        auto simulator_find_last_unsolved_node_index = [&]() -> int64_t {
            int64_t i = int64_t(simulator.size()) - 1;
            while (i >= 0) {
                auto &node = simulator[size_t(i)];
                auto &bubble = node.bubble();
                if (bubble.op() != Bubble::Parameter) return i;
                --i;
            }
            return -1;
        };

        /**
         * \brief find last node ref the given node `ref`
         * \param ref give node
         * \return found index
         * \note return -1 if failed
         */
        auto simulator_find_last_ref_node_index = [&](Node ref) -> int64_t {
            int64_t i = int64_t(simulator.size()) - 1;
            while (i >= 0) {
                auto &node = simulator[size_t(i)];
                auto &refs = map_node_refs[node];
                if (refs.find(ref) != refs.end()) return i;
                --i;
            }
            return -1;
        };

        for (auto &node : outputs) simulator_push(node);

        // TODO: checking inplace operator converting
        // TODO: check if there are some repeating computing brach
        while (unsolved_node_count) {
            auto node = simulator.back();
            auto bubble = node.bubble();
            // case1: check if node are same node
            auto i = simulator.size() - 1;
            auto it = working_nodes.find(node);
            if (it != working_nodes.end()) {
                if (it->second < i) {
                    block.instructions.push_back(instruction::Stack::push(int(it->second)));
                    simulator_pop();
                    continue;
                }
            }

            // case2-0: use Const
            if (bubble.op() == Bubble::Const) {
                int data_index = push_data_sagment(node);
                block.instructions.push_back(std::make_shared<DataSegmentInstruction>(data_index));
                simulator_pop();
                continue;
            }

            // case2-1: use Variable
            if (bubble.op() == Bubble::Variable) {
                TS_LOG_ERROR << "Not support " << Bubble::Variable << " in this version" << eject;
            }

            // case2: save input nodes, move last unsolved node to top
            if (bubble.op() == Bubble::Parameter) {
                auto j = simulator_find_last_unsolved_node_index();
                TS_AUTO_CHECK(j >= 0);
                block.instructions.push_back(instruction::Stack::swap(int(i), int(j)));
                simulator_swap(int(i), int(j));
                continue;
            }

            // case3: check if this node will be compute later, if true, then swap it's son to top
            auto last_ref_node_index = simulator_find_last_ref_node_index(node);
            if (last_ref_node_index >= 0) {
                auto j = last_ref_node_index;
                block.instructions.push_back(instruction::Stack::swap(int(i), int(j)));
                simulator_swap(int(i), int(j));
                continue;
            }

            // case4: found a node need to be compute. query operator
            auto operator_instructions = convert_operator_instruction(node);
            for (auto inst_it = operator_instructions.rbegin(); inst_it != operator_instructions.rend(); ++inst_it) {
                block.instructions.push_back(*inst_it);
            }
            simulator_pop();
            for (auto &input : node.inputs()) simulator_push(input);
        }
        // check inputs
        set<Node> have_inputs(inputs.begin(), inputs.end());
        for (auto &node : simulator) {
            if (have_inputs.find(node) == have_inputs.end()) {
                TS_LOG_ERROR << "Can not access input node: " << node.str() << eject;
            }
        }

        // build inputs
        // -.1 check if inputs satisfied
        bool satisfied = false;
        if (simulator.size() == inputs.size()) {
            satisfied = true;
            for (size_t i = 0; i < simulator.size(); ++i) {
                if (simulator[i] != inputs[i]) {
                    satisfied = false;
                    break;
                }
            }
        }
        if (!satisfied) {
            map<Node, size_t> working_input_nodes;
            for (auto &node : inputs) working_input_nodes.insert(std::make_pair(node, working_input_nodes.size()));
            block.instructions.push_back(instruction::Stack::erase(0, -int(simulator.size())));
            for (auto it = simulator.rbegin(); it != simulator.rend(); ++it) {
                block.instructions.push_back(instruction::Stack::push(int(working_input_nodes[*it])));
            }
            simulator.clear();
            simulator.insert(simulator.begin(), inputs.begin(), inputs.end());
        }

        // reduce
        // 删除冗余的push，是否有必要，push的成本很低
        // inplace operator 是不是可以检测operator，如果是inplace操作，就把push换成clone。或者不支持inplace操作，最简单了。
        // 思考一下怎么处理额，可以在图的编译阶段，如果支持inplace操作，就插入一个copy节点。

        // reverse
        std::reverse(block.instructions.begin(), block.instructions.end());

        return block;
    }

    /**
     *
     * @param [in] node
     * @param [out] const_node
     * @return true if node if const node
     */
    static bool run_const_node(const Node &node, Node &const_node,
            std::unordered_map<Node, Node> &ready_const,
            std::unordered_map<Node, Node> &ready_nonconst) {
        // check ready nonconst
        auto nonconst_it = ready_nonconst.find(node);
        if (nonconst_it != ready_nonconst.end()) {
            const_node = nonconst_it->second;
            return false;
        }
        // check ready const
        auto const_it = ready_const.find(node);
        if (const_it != ready_const.end()) {
            const_node = const_it->second;
            return true;
        }

        // check endpoints
        if(node->op() == Bubble::Variable) {
            TS_LOG_ERROR << "Not support " << Bubble::Variable << " in this version" << eject;
        } else if (node->op() == Bubble::Const) {
            const_node = node;
            ready_const.insert(std::make_pair(node, node));
            return true;
        } else if (node->op() == Bubble::Parameter) {
            const_node = node;
            ready_nonconst.insert(std::make_pair(node, node));
            return false;
        }

        // check if each input is const or new
        std::vector<Tensor> const_inputs;
        std::vector<Node> const_input_nodes;
        bool input_are_const = true;
        bool build_new_node = false;
        for (const auto &input : node.inputs()) {
            auto const_input = input;
            if (!run_const_node(input, const_input, ready_const, ready_nonconst)) {
                input_are_const = false;
            } else {
                const_inputs.emplace_back(const_input->get(name::value));
            }
            build_new_node = build_new_node || const_input != input;
            const_input_nodes.emplace_back(const_input);
        }

        if (!input_are_const) {
            if (!build_new_node) {
                const_node = node;
            } else {
                const_node = ts::bubble::bubble(node.bubble());
                Node::Link(const_node, const_input_nodes);
            }
            ready_nonconst.insert(std::make_pair(node, const_node));
            return false;
        }

        // set const node
        Tensor const_data = intime::run(node.bubble(), const_inputs);
        const_node = ts::bubble::data(node.bubble().name(), const_data);
        ready_const.insert(std::make_pair(node, const_node));
        // walking on origin graph, no need to map new node
        // ready_const.insert(std::make_pair(const_node, const_node));

        return true;
    }


    void Compiler::run_const_nodes(const std::vector<Node> &nodes, std::vector<Node> &const_nodes) {
        const_nodes.clear();
        std::unordered_map<Node, Node> ready_const;
        std::unordered_map<Node, Node> ready_nonconst;
        for (auto node : nodes) {
            run_const_node(node, node, ready_const, ready_nonconst);
            const_nodes.emplace_back(node);
        }
    }

    InstructionBlock Compiler::compile(const std::vector<Node> &raw_inputs, const std::vector<Node> &outputs) {
        return compile(raw_inputs, outputs, "");
    }
}
