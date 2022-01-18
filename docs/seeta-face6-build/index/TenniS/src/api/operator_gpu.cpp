//
// Created by kier on 19-5-18.
//

#include "declare_operator.h"

#ifdef TS_USE_CUDA

#include "api/operator_gpu.h"
#include "kernels/gpu/cuda_context.h"

using namespace ts;

void *ts_OperatorContext_cuda_stream(ts_OperatorContext *context) {
    TRY_HEAD
        if (context->device->computing_device != GPU)
            TS_LOG_ERROR << "The computing device is not gpu! " << eject;
        CUDAContextHandle* handle = reinterpret_cast<CUDAContextHandle*>(context->device->handle);
        if(handle == nullptr)
            TS_LOG_ERROR << "The CUDAContextHandle is null! " << eject;
        auto cuda_stream = handle->stream();
    RETURN_OR_CATCH(cuda_stream, nullptr)
}

#else

using namespace ts;

void *ts_OperatorContext_cuda_stream(ts_OperatorContext *context) {
    TRY_HEAD
        TS_LOG_ERROR << "TensorStack not compiled with TS_USE_CUDA. Can not get CUDA stream." << eject;
    RETURN_OR_CATCH(nullptr, nullptr)
}

#endif
