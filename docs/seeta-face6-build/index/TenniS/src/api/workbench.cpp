//
// Created by kier on 2019/3/16.
//

#include <api/workbench.h>

#include "declare_workbench.h"
#include "declare_module.h"
#include "declare_tensor.h"
#include "declare_image_filter.h"
#include "declare_program.h"

using namespace ts;

ts_Workbench *ts_Workbench_Load(const ts_Module *module, const ts_Device *device) {
    TRY_HEAD
    if (!module) throw Exception("NullPointerException: @param: 1");
    if (!device) throw Exception("NullPointerException: @param: 2");
    std::unique_ptr<ts_Workbench> workbench(new ts_Workbench(
            Workbench::Load(module->pointer, ComputingDevice(device->type, device->id))));
    RETURN_OR_CATCH(workbench.release(), nullptr)
}

void ts_free_Workbench(const ts_Workbench *workbench) {
    TRY_HEAD
    delete workbench;
    TRY_TAIL
}

ts_Workbench *ts_Workbench_clone(ts_Workbench *workbench) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Workbench> dolly(new ts_Workbench((*workbench)->clone()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Workbench_input(ts_Workbench *workbench, int32_t i, const ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    (*workbench)->input(i, **tensor);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_input_by_name(ts_Workbench *workbench, const char *name, const ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!name) throw Exception("NullPointerException: @param: 2");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    (*workbench)->input(name, **tensor);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_run(ts_Workbench *workbench) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    (*workbench)->run();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_output(ts_Workbench *workbench, int32_t i, ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    **tensor = (*workbench)->output(i);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_output_by_name(ts_Workbench *workbench, const char *name, ts_Tensor *tensor) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    if (!name) throw Exception("NullPointerException: @param: 2");
    if (!tensor) throw Exception("NullPointerException: @param: 3");
    **tensor = (*workbench)->output(name);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_set_computing_thread_number(ts_Workbench *workbench, int32_t number) {
    TRY_HEAD
    if (!workbench) throw Exception("NullPointerException: @param: 1");
    (*workbench)->runtime().set_computing_thread_number(number);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_bind_filter(ts_Workbench *workbench, int32_t i, const ts_ImageFilter *filter) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!filter) throw Exception("NullPointerException: @param: 3");
        (*workbench)->bind_filter(i, filter->pointer);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_bind_filter_by_name(ts_Workbench *workbench, const char *name, const ts_ImageFilter *filter) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!name) throw Exception("NullPointerException: @param: 2");
        if (!filter) throw Exception("NullPointerException: @param: 3");
        (*workbench)->bind_filter(name, filter->pointer);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_Workbench *ts_new_Workbench(const ts_Device *device) {
    TRY_HEAD
        //if (!device) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Workbench> workbench(
                device
                ? new ts_Workbench(ComputingDevice(device->type, device->id))
                : new ts_Workbench(ComputingDevice())
        );
    RETURN_OR_CATCH(workbench.release(), nullptr)
}

ts_bool ts_Workbench_setup_context(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        (*workbench)->setup_context();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_setup(ts_Workbench *workbench, const ts_Program *program) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!program) throw Exception("NullPointerException: @param: 2");
        (*workbench)->setup(program->pointer);
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_Program *ts_Workbench_compile(ts_Workbench *workbench, const ts_Module *module) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!module) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Program> program(new ts_Program(
                (*workbench)->compile(module->pointer)));
    RETURN_OR_CATCH(program.release(), nullptr)
}

ts_bool ts_Workbench_setup_device(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        (*workbench)->setup_device();
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_setup_runtime(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        (*workbench)->setup_runtime();
    RETURN_OR_CATCH(ts_true, ts_false)
}

int32_t ts_Workbench_input_count(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        auto result = (*workbench)->input_count();
    RETURN_OR_CATCH(result, 0)
}

int32_t ts_Workbench_output_count(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        auto result = (*workbench)->output_count();
    RETURN_OR_CATCH(result, 0)
}

ts_Workbench *ts_Workbench_Load_v2(const ts_Module *module, const ts_Device *device, const char *options) {
    TRY_HEAD
        if (!module) throw Exception("NullPointerException: @param: 1");
        if (!device) throw Exception("NullPointerException: @param: 2");
        if (!options) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Workbench> workbench(new ts_Workbench(
                Workbench::Load(module->pointer, ComputingDevice(device->type, device->id), options)
                ));
    RETURN_OR_CATCH(workbench.release(), nullptr)
}

ts_Program *ts_Workbench_compile_v2(ts_Workbench *workbench, const ts_Module *module, const char *options) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!module) throw Exception("NullPointerException: @param: 2");
        if (!options) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Program> program(new ts_Program(
                (*workbench)->compile(module->pointer, options)
                ));
    RETURN_OR_CATCH(program.release(), nullptr)
}

ts_bool ts_Workbench_run_hook(ts_Workbench *workbench, const char **node_names, int32_t len) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!node_names) throw Exception("NullPointerException: @param: 2");
        (*workbench)->run_hook(std::vector<std::string>(node_names, node_names + len));
    RETURN_OR_CATCH(ts_true, ts_false)
}

ts_bool ts_Workbench_set_operator_param(ts_Workbench *workbench, const char *node_name, const char *param,
                                        const ts_Tensor *value) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        if (!node_name) throw Exception("NullPointerException: @param: 2");
        if (!param) throw Exception("NullPointerException: @param: 3");
        if (!value) throw Exception("NullPointerException: @param: 4");
        (*workbench)->set_operator_param(node_name, param, **value);
    RETURN_OR_CATCH(ts_true, ts_false)
}

const char *ts_Workbench_summary(ts_Workbench *workbench) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        auto summary = (*workbench)->summary().c_str();
    RETURN_OR_CATCH(summary, nullptr)
}

ts_bool ts_Workbench_set_cpu_mode(ts_Workbench *workbench, ts_CpuPowerMode mode) {
    TRY_HEAD
        if (!workbench) throw Exception("NullPointerException: @param: 1");
        auto set = (*workbench)->set_cpu_power_mode(CpuEnable::CpuPowerMode(mode));
    RETURN_OR_CATCH(ts_bool(set), ts_false)
}
