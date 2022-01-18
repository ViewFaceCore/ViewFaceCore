//
// Created by kier on 2019/3/16.
//

#include <api/tensor.h>

#include "declare_tensor.h"

#include "core/tensor_builder.h"

using namespace ts;

ts_Tensor *ts_new_Tensor(const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data) {
    TRY_HEAD
    if (shape == nullptr) shape_len = 0;
    std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());

    if (data == nullptr) {
        **tensor = Tensor(DTYPE(dtype), Shape(shape, shape + shape_len));
        return tensor.release();
    }

    switch (dtype) {
        default:
            TS_LOG_ERROR << "Not support dtype: " << dtype << eject;
            break;
#define DECLARE_TENSOR_BUILD(api_dtype, type) \
        case api_dtype: \
            **tensor = tensor::build(DTYPE(dtype), Shape(shape, shape + shape_len), \
                    reinterpret_cast<const type*>(data)); \
            break;

        DECLARE_TENSOR_BUILD(TS_INT8, int8_t)
        DECLARE_TENSOR_BUILD(TS_UINT8, uint8_t)
        DECLARE_TENSOR_BUILD(TS_INT16, int16_t)
        DECLARE_TENSOR_BUILD(TS_UINT16, uint16_t)
        DECLARE_TENSOR_BUILD(TS_INT32, int32_t)
        DECLARE_TENSOR_BUILD(TS_UINT32, uint32_t)
        DECLARE_TENSOR_BUILD(TS_INT64, int64_t)
        DECLARE_TENSOR_BUILD(TS_UINT64, uint64_t)
        DECLARE_TENSOR_BUILD(TS_FLOAT32, float)
        DECLARE_TENSOR_BUILD(TS_FLOAT64, double)

#undef DECLARE_TENSOR_BUILD
        case TS_CHAR8:  // this is for trying build string
        {
            Tensor tensor_string = tensor::build(INT8, Shape(shape, shape + shape_len),
                                                             reinterpret_cast<const int8_t*>(data));
            tensor_string = tensor::cast(CHAR8, tensor_string);
            **tensor = tensor_string;
            break;
        }
    }

    RETURN_OR_CATCH(tensor.release(), nullptr)
}

void ts_free_Tensor(const ts_Tensor *tensor) {
    TRY_HEAD
    delete tensor;
    TRY_TAIL
}

const int32_t *ts_Tensor_shape(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    const int32_t *shape = (*tensor)->sizes().data();
    RETURN_OR_CATCH(shape, nullptr)
}

int32_t ts_Tensor_shape_size(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto size = int32_t((*tensor)->dims());
    RETURN_OR_CATCH(size, 0)
}

ts_DTYPE ts_Tensor_dtype(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto dtype = ts_DTYPE((*tensor)->proto().dtype());
    RETURN_OR_CATCH(dtype, TS_VOID)
}

void *ts_Tensor_data(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    auto data = (*tensor)->data();
    RETURN_OR_CATCH(data, nullptr)
}

ts_Tensor *ts_Tensor_clone(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->clone_shared()));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_sync_cpu(ts_Tensor *tensor) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    **tensor = (*tensor)->view(MemoryDevice(CPU));
    RETURN_OR_CATCH(true, false)
}

ts_Tensor *ts_Tensor_cast(ts_Tensor *tensor, ts_DTYPE dtype) {
    TRY_HEAD
    if (!tensor) throw Exception("NullPointerException: @param: 1");
    std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(tensor::cast(DTYPE(dtype), **tensor)));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_Tensor_reshape(ts_Tensor *tensor, const int32_t *shape, int32_t shape_len) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->reshape(Shape(shape, shape + shape_len))));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_Tensor_view_in_flow(ts_Tensor *tensor, ts_InFlow in_flow) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->view(Tensor::InFlow(in_flow))));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *
ts_new_Tensor_in_flow(ts_InFlow in_flow, const int32_t *shape, int32_t shape_len, ts_DTYPE dtype, const void *data) {
    TRY_HEAD
        if (shape == nullptr) shape_len = 0;
        std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());

        **tensor = Tensor(Tensor::InFlow(in_flow), DTYPE(dtype), Shape(shape, shape + shape_len));

        if (data != nullptr) {
            auto bytes = (*tensor)->proto().type_bytes() * (*tensor)->count();
            memcpy((*tensor)->data(), (*tensor)->device(), bytes,
                   data, MemoryDevice(CPU), bytes);
        }

    RETURN_OR_CATCH(tensor.release(), nullptr)
}

ts_Tensor *ts_Tensor_field(ts_Tensor *tensor, int32_t index) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor((*tensor)->field(index)));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_packed(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        if (!(*tensor)->packed()) return ts_false;
    RETURN_OR_CATCH(ts_true, ts_false)
}

int32_t ts_Tensor_fields_count(ts_Tensor *tensor) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        auto fields_count = int32_t((*tensor)->fields_count());
    RETURN_OR_CATCH(fields_count, 0)
}

ts_Tensor *ts_Tensor_pack(ts_Tensor **fields, int32_t count) {
    TRY_HEAD
        if (!fields) throw Exception("NullPointerException: @param: 1");
        std::vector<Tensor> ts_fields;
        for (int i = 0; i < count; ++i) {
            if (!fields[i]) throw Exception("NullPointerException: @param: fields[" + std::to_string(i) + "]");
            ts_fields.emplace_back(**fields[i]);
        }
        std::unique_ptr<ts_Tensor> tensor(new ts_Tensor());
        (*tensor)->pack(ts_fields);
    RETURN_OR_CATCH(tensor.release(), nullptr)
}

ts_Tensor *ts_Tensor_slice(ts_Tensor *tensor, int32_t i) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                (*tensor)->slice(i)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_Tensor_slice_v2(ts_Tensor *tensor, int32_t beg, int32_t end) {
    TRY_HEAD
        if (!tensor) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                (*tensor)->slice(beg, end)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_bool ts_Tensor_save(const char *path, const ts_Tensor *tensor) {
    TRY_HEAD
        if (!path) throw Exception("NullPointerException: @param: 1");
        if (!tensor) throw Exception("NullPointerException: @param: 2");
        tensor::save(path, **tensor);
    RETURN_OR_CATCH(true, false)
}

ts_Tensor *ts_Tensor_load(const char *path) {
    TRY_HEAD
        if (!path) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                tensor::load(path)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

