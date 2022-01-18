//
// Created by kier on 19-5-13.
//

#include "api/intime.h"

#include "declare_tensor.h"

#include "frontend/intime.h"

using namespace ts;

ts_Tensor *ts_intime_transpose(const ts_Tensor *x, const int32_t *permute, int32_t len) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!permute) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::transpose(**x, std::vector<int32_t>(permute, permute + len))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_sigmoid(const ts_Tensor *x) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::sigmoid(**x)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_gather(const ts_Tensor *x, const ts_Tensor *indices, int32_t axis) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!indices) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::gather(**x, **indices, axis)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_concat(const ts_Tensor *const *x, int32_t len, int32_t dim) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        std::vector<Tensor> ts_inputs;
        for (int i = 0; i < len; ++i) {
            if (!x[i]) throw Exception("NullPointerException: @param: x[" + std::to_string(i) + "]");
            ts_inputs.emplace_back(**x[i]);
        }
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::concat(ts_inputs, dim)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_softmax(const ts_Tensor *x, int32_t dim, ts_bool smooth) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::softmax(**x, dim, bool(smooth))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_pad(const ts_Tensor *x, const ts_Tensor *padding, float padding_value) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!padding) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::pad(**x, **padding, padding_value)
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_cast(const ts_Tensor *x, ts_DTYPE dtype) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::cast(**x, ts::DTYPE(dtype))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_resize2d(const ts_Tensor *x, const ts_Tensor *size, int32_t method) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!size) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::resize2d(**x, **size, ts::desc::ResizeType(method))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_affine_sample2d(const ts_Tensor *x, const ts_Tensor *size, const ts_Tensor *affine,
                                     int32_t dim, float outer_value, int32_t method) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!size) throw Exception("NullPointerException: @param: 2");
        if (!affine) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::affine_sample2d(**x, **size, **affine, dim, outer_value, ts::desc::ResizeType(method))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

ts_Tensor *ts_intime_affine_on_sample2d(const ts_Tensor *x, const ts_Tensor *size, const ts_Tensor *affine, int32_t dim,
                                        int32_t method) {
    TRY_HEAD
        if (!x) throw Exception("NullPointerException: @param: 1");
        if (!size) throw Exception("NullPointerException: @param: 2");
        if (!affine) throw Exception("NullPointerException: @param: 3");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::affine_on_sample2d(**x, **size, **affine, dim, ts::desc::ResizeType(method))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}

int64_t ts_intime_memcpy(
        ts_Tensor *dst_desc, void *dst_ptr, int64_t dst_shift,
        const ts_Tensor *src_desc, const void *src_ptr, int64_t src_shift,
        int64_t size) {
    TRY_HEAD
        if (!dst_desc) throw Exception("NullPointerException: @param: 1");
        if (!src_desc) throw Exception("NullPointerException: @param: 2");
        auto copied = intime::memcpy(
                **dst_desc, dst_ptr, dst_shift,
                **src_desc, src_ptr, src_shift,
                size);
    RETURN_OR_CATCH(copied, 0);
}

ts_Tensor *ts_intime_matmul(const ts_Tensor *A, const ts_Tensor *B, ts_bool transpose) {
    TRY_HEAD
        if (!A) throw Exception("NullPointerException: @param: 1");
        if (!B) throw Exception("NullPointerException: @param: 2");
        std::unique_ptr<ts_Tensor> dolly(new ts_Tensor(
                intime::matmul(**A, **B, bool(transpose))
        ));
    RETURN_OR_CATCH(dolly.release(), nullptr)
}
