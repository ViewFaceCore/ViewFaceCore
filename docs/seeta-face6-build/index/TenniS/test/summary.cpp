//
// Created by kier on 2019/11/23.
//

#include <algorithm>
#include "global/device_admin.h"
#include "global/fp16_operator_factory.h"
#include "global/hard_allocator.h"
#include "global/hard_converter.h"
#include "global/memory_device.h"
#include "global/operator_factory.h"
#include "global/shape_inferer_factory.h"

#include <iterator>

template <typename T>
std::ostream &plot_set(std::ostream &out, const std::set<T> &values, const std::function<std::string(const T &)> &str) {
    static const int WIDTH = 80;
    int cursor = 0;

    auto push = [&](const std::string &msg) {
        cursor += msg.length();
        out << msg;
    };

    push("[");
    bool first = true;
    for (auto &x : values) {
        if (first) {
            first = false;
        } else {
            push(", ");
        }
        auto next = str(x);
        if (cursor + next.length() + 2 > WIDTH) {
            cursor = 0;
            out << std::endl;
            push(" ");
        }
        push(next);
    }
    push("]");
    return out;
}

std::string copy(const std::string &x) {
    return x;
}

std::string dst_src(const std::pair<std::string, std::string> &x) {
    return x.first + " <- " + x.second;
}

std::string computing_memory(const std::pair<std::string, std::string> &x) {
    return x.first + ":" + x.second;
}

std::map<std::string, std::set<std::string>> set2map(const std::set<std::pair<std::string, std::string>> &x) {
    std::map<std::string, std::set<std::string>> y;
    for (auto &item : x) {
        auto &key = item.first;
        auto &value = item.second;

        auto it = y.find(key);
        if (it == y.end()) {
            y.insert(std::make_pair(key, std::set<std::string>({value})));
        } else {
            it->second.insert(value);
        }
    }
    return y;
}

std::set<std::string> set2set_value(const std::set<std::pair<std::string, std::string>> &x) {
    std::set<std::string> y;
    for (auto &item : x) {
        auto &value = item.second;

        y.insert(value);
    }
    return y;
}

int main() {
    std::cout << "== Device admin ==" << std::endl;
    auto device_admin = ts::DeviceAdmin::AllKeys();
    plot_set<std::string>(std::cout, device_admin, copy) << std::endl;
    std::cout << "Count: " << device_admin.size() << std::endl;
    std::cout << std::endl;

    std::cout << "== Hard allocator ==" << std::endl;
    auto hard_allocator = ts::HardAllocator::AllKeys();
    plot_set<std::string>(std::cout, hard_allocator, copy) << std::endl;
    std::cout << "Count: " << hard_allocator.size() << std::endl;
    std::cout << std::endl;

    std::cout << "== Hard converter ==" << std::endl;
    auto hard_converter = ts::HardConverter::AllKeys();
    plot_set<std::pair<std::string, std::string>>(std::cout, hard_converter, dst_src) << std::endl;
    std::cout << "Count: " << hard_converter.size() << std::endl;
    std::cout << std::endl;

    std::cout << "== Computing Device:Memory device ==" << std::endl;
    auto computing_device = ts::ComputingMemory::AllItems();
    plot_set<std::pair<std::string, std::string>>(std::cout, computing_device, computing_memory) << std::endl;
    std::cout << "Count: " << computing_device.size() << std::endl;
    std::cout << std::endl;

    std::cout << "== Shape infer ==" << std::endl;
    auto shape_inferer = ts::ShapeInferer::AllKeys();
    plot_set<std::string>(std::cout, shape_inferer, copy) << std::endl;
    std::cout << "Count: " << shape_inferer.size() << std::endl;
    std::cout << std::endl;

    std::cout << "== Operator ==" << std::endl;
    auto operators = ts::OperatorCreator::AllKeys();

    auto all_ops = set2set_value(operators);
    std::cout << "All operators count: " << all_ops.size() << std::endl;

    auto device_operators = set2map(operators);
    for (auto &dev_ops : device_operators) {
        auto &device = dev_ops.first;
        auto &ops = dev_ops.second;

        std::cout << "-- " << device << " --" << std::endl;
        plot_set<std::string>(std::cout, ops, copy) << std::endl;
        std::cout << "Count: " << ops.size() << std::endl;
        std::cout << std::endl;
    }

    std::cout << "== Float16 operator ==" << std::endl;
    auto fp16_operators = ts::Fp16OperatorCreator::AllKeys();

    auto fp16_device_operators = set2map(fp16_operators);
    for (auto &dev_ops : fp16_device_operators) {
        auto &device = dev_ops.first;
        auto &ops = dev_ops.second;

        std::cout << "-- " << device << " --" << std::endl;
        plot_set<std::string>(std::cout, ops, copy) << std::endl;
        std::cout << "Count: " << ops.size() << std::endl;
        std::cout << std::endl;
    }

    std::cout << "== All Attributes ==" << std::endl;
    std::set<std::string> all_attr_set;
    for (auto &device_op : operators) {
        auto op = ts::OperatorCreator::Create(device_op.first, device_op.second);
        auto fields = op->list_all_fields();
        all_attr_set.insert(fields.begin(), fields.end());
    }
    plot_set<std::string>(std::cout, all_attr_set, copy) << std::endl;
    std::cout << "Count: " << all_attr_set.size() << std::endl;
    std::cout << std::endl;



    std::cout << "-- == ++ Warning part ++ == --" << std::endl;
    auto &cpu_op_set = device_operators["cpu"];

    std::cout << "== No shape infer ==" << std::endl;
    std::set<std::string> no_shape_infer;
    std::set_difference(
            cpu_op_set.begin(), cpu_op_set.end(),
            shape_inferer.begin(), shape_inferer.end(),
            std::inserter(no_shape_infer, no_shape_infer.begin()));
    plot_set<std::string>(std::cout, no_shape_infer, copy) << std::endl;
    std::cout << "Count: " << no_shape_infer.size() << std::endl;
    std::cout << std::endl;

    for (auto &dev_ops : device_operators) {
        auto &device = dev_ops.first;
        auto &ops = dev_ops.second;
        if (device == "cpu") continue;

        std::cout << "== Device " << device << " not support ==" << std::endl;

        std::set<std::string> not_supported;
        std::set_difference(
                cpu_op_set.begin(), cpu_op_set.end(),
                ops.begin(), ops.end(),
                std::inserter(not_supported, not_supported.begin()));
        plot_set<std::string>(std::cout, not_supported, copy) << std::endl;
        std::cout << "Count: " << not_supported.size() << std::endl;
        std::cout << std::endl;
    }

    return 0;
}