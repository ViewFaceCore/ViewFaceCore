//
// Created by kier on 2019/5/7.
//

#include <api/program.h>

#include "declare_module.h"
#include "declare_tensor.h"
#include "declare_program.h"

using namespace ts;

ts_Program *ts_Program_Compile(const ts_Module *module, const ts_Device *device) {
    TRY_HEAD
    if (!module) throw Exception("NullPointerException: @param: 1");
    if (!device) throw Exception("NullPointerException: @param: 2");
    std::unique_ptr<ts_Program> program(new ts_Program(
            Program::Compile(module->pointer, ComputingDevice(device->type, device->id))));
    RETURN_OR_CATCH(program.release(), nullptr)
}

void ts_free_Program(const ts_Program *program) {
    TRY_HEAD
    delete program;
    TRY_TAIL
}

ts_Program *ts_Program_clone(ts_Program *program) {
    TRY_HEAD
        if (!program) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Program> dolly(new ts_Program((*program)->clone()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

int32_t ts_Program_input_count(ts_Program *program) {
    TRY_HEAD
        if (!program) throw Exception("NullPointerException: @param: 1");
        auto result = (*program)->input_count();
    RETURN_OR_CATCH(result, 0)
}

int32_t ts_Program_output_count(ts_Program *program) {
    TRY_HEAD
        if (!program) throw Exception("NullPointerException: @param: 1");
        auto result = (*program)->output_count();
    RETURN_OR_CATCH(result, 0)
}

ts_Program *ts_Program_Compile_v2(const ts_Module *module, const ts_Device *device, const char *options) {
    TRY_HEAD
        if (!module) throw Exception("NullPointerException: @param: 1");
        if (!device) throw Exception("NullPointerException: @param: 2");
        if (!options) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Program> program(new ts_Program(
                Program::Compile(module->pointer, ComputingDevice(device->type, device->id), options)
                ));
    RETURN_OR_CATCH(program.release(), nullptr)
}

ts_bool
ts_Program_set_operator_param(ts_Program *program, const char *node_name, const char *param, const ts_Tensor *value) {
    TRY_HEAD
        if (!program) throw Exception("NullPointerException: @param: 1");
        if (!node_name) throw Exception("NullPointerException: @param: 2");
        if (!param) throw Exception("NullPointerException: @param: 3");
        if (!value) throw Exception("NullPointerException: @param: 4");
        (*program)->set_operator_param(node_name, param, **value);
    RETURN_OR_CATCH(ts_true, ts_false)
}
