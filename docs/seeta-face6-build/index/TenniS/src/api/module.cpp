//
// Created by kier on 2019/3/16.
//

#include <api/module.h>

#include "declare_module.h"

using namespace ts;

ts_Module *ts_Module_Load(const char *filename, ts_SerializationFormat format) {
    TRY_HEAD
    if (!filename) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Module> module(new ts_Module(
            Module::Load(filename, Module::SerializationFormat(format))));
    RETURN_OR_CATCH(module.release(), nullptr)
}

void ts_free_Module(const ts_Module *module) {
    TRY_HEAD
    delete module;
    TRY_TAIL
}

class CStreamReader : public ts::StreamReader {
public:
    CStreamReader(void *obj, ts_stream_read *reader)
        : m_obj(obj), m_reader(reader) {}

    size_t read(void *buf, size_t len) override {
        auto ret = m_reader(m_obj, reinterpret_cast<char *>(buf), uint64_t(len));
        return size_t(ret);
    }

private:
    void *m_obj;
    ts_stream_read *m_reader;
};

ts_Module *ts_Module_LoadFromStream(void *obj, ts_stream_read *reader, ts_SerializationFormat format) {
    TRY_HEAD
    if (!obj) throw Exception("NullPointerException: @param: 1");
    if (!reader) throw Exception("NullPointerException: @param: 2");
    CStreamReader stream_reader(obj, reader);
    std::unique_ptr<ts_Module> module(new ts_Module(
            Module::Load(stream_reader, Module::SerializationFormat(format))));
    RETURN_OR_CATCH(module.release(), nullptr)
}

ts_Module *ts_Module_translate(const ts_Module *module, const ts_Device *device, const char *options) {
    TRY_HEAD
        if (!module) throw Exception("NullPointerException: @param: 1");
        if (!device) throw Exception("NullPointerException: @param: 2");
        if (!options) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Module> translated_module(new ts_Module(
                Module::Translate((*module).pointer, ComputingDevice(device->type, device->id), options)
                ));
    RETURN_OR_CATCH(translated_module.release(), nullptr)
}

ts_Module *ts_Module_Fusion(const ts_Module *in, int32_t in_out_slot, const ts_Module *out, int32_t out_in_slot) {
    TRY_HEAD
        if (!in) throw Exception("NullPointerException: @param: 1");
        if (!out) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Module> module(new ts_Module(
                Module::Fusion({(*in).pointer, (*out).pointer}, {{0, in_out_slot, 1, out_in_slot}})
        ));
    RETURN_OR_CATCH(module.release(), nullptr)
}
