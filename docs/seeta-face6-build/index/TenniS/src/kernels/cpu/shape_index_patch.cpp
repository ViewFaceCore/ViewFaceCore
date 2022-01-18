#include <kernels/cpu/shape_index_patch.h>
#include <core/tensor_builder.h>
#include <memory>
#include <global/operator_factory.h>
#include <backend/name.h>
#include <core/device.h>
#include <utils/assert.h>

namespace ts {
    namespace cpu {
        template<typename T>
        static inline void cpu_sample_compute_run(const Tensor &x, const Tensor &pos,
                                                  const Size2D &origin_patch, const Size2D &origin,
                                                  Tensor &out) {
            auto &feat_blob = x;
            auto &pos_blob = pos;


            int feat_h = feat_blob.size(2);
            int feat_w = feat_blob.size(3);

            int feat_patch_h = out.size(2);
            int feat_patch_w = out.size(4);

            const int num = feat_blob.size(0);
            const int channels = feat_blob.size(1);
            const float r_h = (feat_patch_h - 1) / 2.0f;
            const float r_w = (feat_patch_w - 1) / 2.0f;
            const int landmark_num = out.size(3);

            HypeShape pos_offset({pos.size(0), pos.size(1)});
            HypeShape feat_offset(feat_blob.sizes());
            HypeShape out_offset(out.sizes());

            // offset
            T *const buff = out.data<T>();
            auto pos_data = pos_blob.data<T>();
            auto feat_data = feat_blob.data<T>();

            T zero = 0;

            for (int i = 0; i < landmark_num; i++) {
                for (int n = 0; n < num; n++) { // x1, y1, ..., xn, yn
                    // coordinate of the first patch pixel, scale to the feature map coordinate
                    const int y = int(pos_data[pos_offset.to_index({n, 2 * i + 1})] * (feat_h - 1) - r_h + 0.5f);
                    const int x = int(pos_data[pos_offset.to_index({n, 2 * i})] * (feat_w - 1) - r_w + 0.5f);

                    for (int c = 0; c < channels; c++) {
                        for (int ph = 0; ph < feat_patch_h; ph++) {
                            for (int pw = 0; pw < feat_patch_w; pw++) {
                                const int y_p = y + ph;
                                const int x_p = x + pw;
                                // set zero if exceed the img bound
                                if (y_p < 0 || y_p >= feat_h || x_p < 0 || x_p >= feat_w)
                                    buff[out_offset.to_index({n, c, ph, i, pw})] = zero;
                                else
                                    buff[out_offset.to_index({n, c, ph, i, pw})] =
                                            feat_data[feat_offset.to_index({n, c, y_p, x_p})];
                            }
                        }
                    }
                }
            }
        }

        void ShapeIndexPatch::sample(const Tensor &x, const Tensor &pos,
                                     const Size2D &origin_patch, const Size2D &origin,
                                     Tensor &out) {
            DTYPE dtype = out.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_sample_compute_run<TYPE>(x, pos, origin_patch, origin, out); break; }
                DECLARE_COMPUTE_RUN(INT8, int8_t);
                DECLARE_COMPUTE_RUN(UINT8, uint8_t);
                DECLARE_COMPUTE_RUN(INT16, int16_t);
                DECLARE_COMPUTE_RUN(UINT16, uint16_t);
                DECLARE_COMPUTE_RUN(INT32, int32_t);
                DECLARE_COMPUTE_RUN(UINT32, uint32_t);
                DECLARE_COMPUTE_RUN(INT64, int64_t);
                DECLARE_COMPUTE_RUN(UINT64, uint64_t);
                DECLARE_COMPUTE_RUN(FLOAT32, float);
                DECLARE_COMPUTE_RUN(FLOAT64, double);
#undef DECLARE_COMPUTE_RUN
                default: {
                    TS_LOG_ERROR << this->op() << " not support data type(" << dtype << "): " << type_str(dtype) << eject;
                    break;
                }
            }
        }
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(ShapeIndexPatch, CPU, name::layer::shape_index_patch())
