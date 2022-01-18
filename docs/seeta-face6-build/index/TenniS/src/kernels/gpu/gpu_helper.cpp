//
// Created by kier on 19-3-30.
//

#include <kernels/gpu/gpu_helper.h>
#include <runtime/runtime.h>

#include "kernels/gpu/gpu_helper.h"
#include "core/tensor_iterator.h"

#include "kernels/gpu/cuda_context.h"
#include "core/device_context.h"
#include "utils/ctxmgr_lite.h"

namespace ts {
    namespace gpu {
        static inline std::shared_ptr<char> cpu_blocked(const std::vector<CpuBlock> &cpu) {
            int shift = 0;
            for (auto &pair : cpu) {
                shift += pair.second;
            }
            std::shared_ptr<char> cpu_memory(new char[shift], std::default_delete<char[]>());
            shift = 0;
            for (auto &pair : cpu) {
                std::memcpy(cpu_memory.get() + shift, pair.first, size_t(pair.second));
                shift += pair.second;
            }
            return std::move(cpu_memory);
        }
        SyncMemory convert_block_to_gpu(
                SyncMemoryController::shared controller,
                const MemoryDevice &device,
                const std::vector<CpuBlock> &cpu,
                const std::vector<void **> &gpu) {
            int shift = 0;
            for (auto &pair : cpu) {
                shift += pair.second;
            }
            std::shared_ptr<char> cpu_memory(new char[shift], std::default_delete<char[]>());
            auto cpu_data = cpu_memory.get();
            auto gpu_memory = controller->alloc(device, size_t(shift));
            auto gpu_data = gpu_memory.data<char>();

            shift = 0;
            for (size_t i = 0; i < cpu.size(); ++i) {
                auto &pair = cpu[i];
                std::memcpy(cpu_data + shift, pair.first, size_t(pair.second));
                *gpu[i] = gpu_data + shift;
                shift += pair.second;
            }

            memcpy(gpu_data, device, shift, cpu_data, MemoryDevice(CPU), shift);

            return std::move(gpu_memory);
        }

        std::pair<SyncMemory, GpuHypeShape> MakeGPUHypeShape(const MemoryDevice &device, const Shape &shape) {
            HypeShape hype_shape(shape);
            GpuHypeShape gpu_shape;
            gpu_shape.dims = int(shape.size());
            auto gpu_memory = convert_block_to_gpu(RuntimeContext::FlowMemory(), device,
                                                   {{(void*)(hype_shape.shape().data()), 4 * int(hype_shape.shape().size())},
                                                    {(void*)(hype_shape.weight().data()), 4 * int(hype_shape.weight().size())},},
                                                   {(void**)(&gpu_shape.shape), (void**)(&gpu_shape.weights)});
            return std::make_pair(std::move(gpu_memory), gpu_shape);
        }

        std::pair<SyncMemory, std::vector<GpuHypeShape>> MakeGPUHypeShape(const MemoryDevice &device, const std::vector<Shape> &shape) {
            return MakeGPUHypeShape(device, shape, {}, {});
        }

        std::pair<SyncMemory, std::vector<GpuHypeShape>> MakeGPUHypeShape(const MemoryDevice &device, const std::vector<Shape> &shape,
                                                                          const std::vector<CpuBlock> &cpu_part,
                                                                          const std::vector<void **> &gpu_part) {
            std::vector<HypeShape> hype_shape_array;
            for (auto &item : shape) {
                hype_shape_array.emplace_back(item);
            }
            std::vector<GpuHypeShape> gpu_shape_array(shape.size());
            std::vector<CpuBlock> cpu;
            std::vector<void **> gpu;
            for (size_t i = 0; i < shape.size(); ++i) {
                auto &hype_shape = hype_shape_array[i];
                auto &gpu_shape = gpu_shape_array[i];
                gpu_shape.dims = int(hype_shape.shape().size());
                cpu.push_back({(void*)(hype_shape.shape().data()), 4 * int(hype_shape.shape().size())});
                cpu.push_back({(void*)(hype_shape.weight().data()), 4 * int(hype_shape.weight().size())});
                gpu.push_back((void**)(&gpu_shape.shape));
                gpu.push_back((void**)(&gpu_shape.weights));
            }
            cpu.insert(cpu.end(), cpu_part.begin(), cpu_part.end());
            gpu.insert(gpu.end(), gpu_part.begin(), gpu_part.end());
            auto gpu_memory = convert_block_to_gpu(RuntimeContext::FlowMemory(), device, cpu, gpu);

            return std::make_pair(std::move(gpu_memory), gpu_shape_array);
        }

        cudaStream_t get_cuda_stream_on_context() {
            auto &context = ctx::ref<DeviceContext>();
            CUDAContextHandle* handle = reinterpret_cast<CUDAContextHandle*>(context.handle);
            if(handle == nullptr)
                TS_LOG_ERROR << "The CUDAContextHandle is null! " << eject;
            auto cuda_stream = handle->stream();
            return cuda_stream;
        }
    }
}
