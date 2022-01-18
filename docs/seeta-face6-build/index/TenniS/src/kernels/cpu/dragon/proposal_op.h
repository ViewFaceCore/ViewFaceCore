/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_
#define DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_

#include "kernels/common/third/dragon.h"

namespace ts {

namespace dragon {

template <class Context>
class ProposalOp final : public Operator<Context> {
 public:
    ProposalOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          strides(OperatorBase::Args<int64_t>("strides")),
          ratios(OperatorBase::Args<float>("ratios")),
          scales(OperatorBase::Args<float>("scales")),
          pre_nms_top_n(OperatorBase::Arg<int64_t>("pre_nms_top_n", 6000)),
          post_nms_top_n(OperatorBase::Arg<int64_t>("post_nms_top_n", 300)),
          nms_thresh(OperatorBase::Arg<float>("nms_thresh", (float)0.7)),
          min_size(OperatorBase::Arg<int64_t>("min_size", 16)),
          min_level(OperatorBase::Arg<int64_t>("min_level", 2)),
          max_level(OperatorBase::Arg<int64_t>("max_level", 5)),
          canonical_level(OperatorBase::Arg<int64_t>("canonical_level", 4)),
          canonical_scale(OperatorBase::Arg<int64_t>("canonical_scale", 224)) {
        temp(anchors_);
        temp(proposals_);
        temp(roi_indices_);
        temp(nms_mask_);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

    template <typename T> void RunWithType(
        const T* scores, const T* bbox_deltas);

 protected:
    vector<int64_t> strides;
    vector<float> ratios, scales;
    int64_t pre_nms_top_n, post_nms_top_n;
    float nms_thresh;
    int64_t min_size, num_images;
    int64_t min_level, max_level, canonical_level, canonical_scale;
    Tensor anchors_, proposals_, roi_indices_, nms_mask_;
};

}  // namespace dragon

}  // namespace ts

#endif  // DRAGON_CONTRIB_RCNN_PROPOSAL_OP_H_