//
// Created by yang on 2020/2/21.
//

#include <map>

#include "api/plugin.h"
#include "global/operator_factory.h"
#include "core/device_context.h"
#include "core/device.h"

#include "declaration.h"

using namespace ts;

using creator_map = std::map<std::pair<DeviceType, std::string>, OperatorCreator::function>;

struct ts_op_creator_map{
    using self = ts_op_creator_map;
    explicit ts_op_creator_map(creator_map map) {this->map=std::move(map);}
    creator_map map;
};

struct ts_device_context{
    using self = ts_device_context;
    explicit ts_device_context(std::shared_ptr<DeviceContext> pointer) : pointer(std::move(pointer)) {}
    std::shared_ptr<DeviceContext> pointer;
};

ts_op_creator_map* ts_plugin_get_creator_map(){
    TRY_HEAD
    auto creator_map = OperatorCreator::GetCreatorFucMap();
    std::unique_ptr<ts_op_creator_map> res(new ts_op_creator_map(creator_map));
    RETURN_OR_CATCH(res.release(), nullptr);
}

void ts_plugin_flush_creator(ts_op_creator_map* creator_map){
    TRY_HEAD
    OperatorCreator::flush(creator_map->map);
    TRY_TAIL
}

void ts_plugin_free_creator_map(ts_op_creator_map* creator_map){
    TRY_HEAD
    creator_map->map.clear();
    delete(creator_map);
    TRY_TAIL
}

ts_device_context* ts_plugin_initial_device_context(const ts_Device *device){
    TRY_HEAD
    std::shared_ptr<DeviceContext> device_context = std::make_shared<DeviceContext>();
    device_context->initialize(ComputingDevice(device->type, device->id));
    DeviceContext::Switch(device_context.get());
    std::unique_ptr<ts_device_context> tdc(new ts_device_context(device_context));
    RETURN_OR_CATCH(tdc.release(), nullptr)
}

void ts_plugin_free_device_context(ts_device_context* device){
    TRY_HEAD
    delete(device);
    TRY_TAIL
}

void ts_plugin_bind_device_context(ts_device_context* device){
    TRY_HEAD
    DeviceContext::Switch(device->pointer.get());
    TRY_TAIL
}


