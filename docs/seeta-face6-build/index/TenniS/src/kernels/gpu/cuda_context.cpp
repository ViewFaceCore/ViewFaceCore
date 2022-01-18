#include "kernels/gpu/cuda_context.h"

#include <kernels/gpu/cuda_context.h>

#include "global/memory_device.h"
#include "utils/static.h"
#include "kernels/gpu/memory_gpu.h"

#include <cuda_runtime.h>

namespace ts {
    void DeviceCuBLASAdminFunction(DeviceHandle **handle, int device_id, DeviceAdmin::Action action)
    {
        auto cuda_error = cudaSetDevice(device_id);
        if (cuda_error != cudaSuccess) {
            TS_LOG_ERROR << "cudaSetDevice(" << device_id << ") failed. Error(" << cuda_error << "): " << cudaGetErrorString(cuda_error) << eject;
        }
        switch (action)
        {
        case ts::DeviceAdmin::INITIALIZATION:
        {
            auto cublas_device_handle = new CUDAContextHandle(device_id);
            *handle = reinterpret_cast<DeviceHandle*>(cublas_device_handle);
            break;
        }
        case ts::DeviceAdmin::FINALIZATION:
        {
            delete reinterpret_cast<CUDAContextHandle*>(*handle);
            *handle = nullptr;
            break;
        }
        case ts::DeviceAdmin::ACTIVATION:
        {
            break;
        }
        case ts::DeviceAdmin::SYNCHRONIZE:
        {
            auto cuda_context = reinterpret_cast<CUDAContextHandle*>(*handle);
            cudaStreamSynchronize(cuda_context->stream());
            break;
        }
        default:
            break;
        }
    }
}
//
//TS_STATIC_ACTION(ts::HardAllocator::Register, ts::CUBLAS, ts::gpu_allocator)
//TS_STATIC_ACTION(ts::HardConverter::Register, ts::CUBLAS, ts::CUBLAS, ts::gpu2gpu_converter)
//TS_STATIC_ACTION(ts::HardConverter::Register, ts::CUBLAS, ts::CPU, ts::cpu2gpu_converter)
//TS_STATIC_ACTION(ts::HardConverter::Register, ts::CPU, ts::CUBLAS, ts::gpu2cpu_converter)
//TS_STATIC_ACTION(ts::ComputingMemory::Register, ts::CUBLAS, ts::GPU)
TS_STATIC_ACTION(ts::DeviceAdmin::Register, ts::GPU, ts::DeviceCuBLASAdminFunction)
