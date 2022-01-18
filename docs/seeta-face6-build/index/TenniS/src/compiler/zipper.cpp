//
// Created by kier on 2018/11/1.
//

#include "compiler/zipper.h"
#include "compiler/option/zipper_option.h"

#include "module/menu.h"

#include <unordered_set>
#include <compiler/argparse.h>

#include "compiler/option/winograd_zipper_option.h"

namespace ts {

    Zipper::Zipper(const ComputingDevice &device)
            : m_device(device) {
    }

    static Node zip_node(const Node &node,
                         std::unordered_map<Node, Node> &ready_map,
                         const ComputingDevice &device,
                         const std::vector<const ZipperOption *> &options) {
        {
            auto ready_it = ready_map.find(node);
            if (ready_it != ready_map.end()) {
                return ready_it->second;
            }
        }

        /**
         * zip this node
         */
        Node zipped_node = node;
        for (auto &option : options) {
            if (option->zip(device, node, zipped_node)) break;
        }

        /**
         * zip inputs
         */
        std::vector<Node> zipped_inputs;
        bool zipped = false;
        for (auto &input : zipped_node.inputs()) {
            auto zipped_input = zip_node(input, ready_map, device, options);
            if (zipped_input != input) zipped = true;
            zipped_inputs.emplace_back(zipped_input);
        }

        /**
         * link if there is new node
         */
        if (zipped) {
            zipped_node = bubble::bubble(zipped_node.bubble());
            Node::Link(zipped_node, zipped_inputs);
        }

        ready_map.insert(std::make_pair(node, zipped_node));

        return zipped_node;
    }

    std::vector<Node> Zipper::zip(const std::vector<Node> &nodes) const {
        if (ctx::get<Graph>() == nullptr) {
            TS_LOG_ERROR << "context:<ts::Graph> needed, but not given." << eject;
        }

        // TODO: add more options
        std::vector<const ZipperOption *> options = GetFullOptions();

        for (auto &option : m_options) {
            options.push_back(option);
        }

        if (options.empty()) return nodes;

        std::vector<Node> zipped_nodes;
        std::unordered_map<Node, Node> ready_map;
        for (auto &node : nodes) {
            zipped_nodes.emplace_back(zip_node(node, ready_map, m_device, options));
        }
        return std::move(zipped_nodes);
    }

    Zipper::Zipper(const ComputingDevice &device, const std::string &params)
        : m_device(device) {
        ArgParser parser;
        //NOTE:Winograd conv was only used on arm device now
#ifdef TS_ON_ARM
        parser.add({"--winograd", "-win"}, {"--no-winograd", "-no-win"}, true);
        parser.parse(params);
        if (parser.get("--winograd")) {
            TS_LOG_STATUS << "Compiling with --winograd";
            m_options.push_back(new Conv2dZipperOption);
        }
#endif
    }

    Zipper::~Zipper() {
        for (auto option : m_options) {
            delete option;
        }
        m_options.clear();
    }
}
