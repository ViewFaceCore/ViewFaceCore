#include "bbox_utils.h"

namespace ts {

namespace dragon {

namespace rcnn {

/******************** Proposal ********************/

template <> void GenerateProposals<float, CPUContext>(
    const int               A,
    const int               feat_h,
    const int               feat_w,
    const int               stride,
    const float             im_h,
    const float             im_w,
    const float             min_box_h,
    const float             min_box_w,
    const float*            scores,
    const float*            bbox_deltas,
    const float*            anchors,
    float*                  proposals,
    CPUContext*             ctx) {
    float* proposal = proposals;
    const int K = feat_h * feat_w;
    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            const float x = (float)w * stride;
            const float y = (float)h * stride;
            // bbox_deltas: [1, A, 4, K]
            const float* bbox_delta = bbox_deltas + h * feat_w + w;
            // scores: [1, A, K]
            const float* score = scores + h * feat_w + w;
            for (int a = 0; a < A; ++a) {
                const float dx = bbox_delta[(a * 4 + 0) * K];
                const float dy = bbox_delta[(a * 4 + 1) * K];
                const float d_log_w = bbox_delta[(a * 4 + 2) * K];
                const float d_log_h = bbox_delta[(a * 4 + 3) * K];
                proposal[0] = x + anchors[a * 4 + 0];
                proposal[1] = y + anchors[a * 4 + 1];
                proposal[2] = x + anchors[a * 4 + 2];
                proposal[3] = y + anchors[a * 4 + 3];
                proposal[4] = BBoxTransform<float>(
                    dx, dy, d_log_w, d_log_h,
                        im_w, im_h, min_box_w, min_box_h,
                            proposal) * score[a * K];
                proposal += 5;
            }
        }
    }
}

template <> void GenerateProposals_v2<float, CPUContext>(
    const int               total_anchors,
    const float             im_h,
    const float             im_w,
    const float             min_box_h,
    const float             min_box_w,
    const float*            scores,
    const float*            bbox_deltas,
    float*                  proposals,
    CPUContext*             ctx) {
    float* proposal = proposals;
    for (int i = 0; i < total_anchors; ++i) {
        // bbox_deltas: [1, 4, total_anchors]
        // scores: [1, total_anchors]
        const float dx = bbox_deltas[i];
        const float dy = bbox_deltas[total_anchors + i];
        const float d_log_w = bbox_deltas[2 * total_anchors + i];
        const float d_log_h = bbox_deltas[3 * total_anchors + i];
        proposal[4] = BBoxTransform<float>(
            dx, dy, d_log_w, d_log_h,
                im_w, im_h, min_box_w, min_box_h,
                    proposal) * scores[i];
        proposal += 5;
    }
}

/******************** NMS ********************/

template <typename T>
T iou(const T A[], const T B[]) {
    if (A[0] > B[2] || A[1] > B[3] ||
        A[2] < B[0] || A[3] < B[1]) return 0;
    const T x1 = std::max(A[0], B[0]);
    const T y1 = std::max(A[1], B[1]);
    const T x2 = std::min(A[2], B[2]);
    const T y2 = std::min(A[3], B[3]);
    const T width = std::max((T)0, x2 - x1 + 1);
    const T height = std::max((T)0, y2 - y1 + 1);
    const T area = width * height;
    const T A_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
    const T B_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
    return area / (A_area + B_area - area);
}

template <> void ApplyNMS<float, CPUContext>(
    const int               num_boxes,
    const int               max_keeps,
    const float             thresh,
    const float*            boxes,
    int*                    keep_indices,
    int&                    num_keep,
    CPUContext*             ctx) {
    int count = 0;
    std::vector<char> is_dead(num_boxes);
    for (int i = 0; i < num_boxes; ++i) is_dead[i] = 0;
    for (int i = 0; i < num_boxes; ++i) {
        if (is_dead[i]) continue;
        keep_indices[count++] = i;
        if (count == max_keeps) break;
        for (int j = i + 1; j < num_boxes; ++j)
            if (!is_dead[j] && iou(&boxes[i * 5],
                                   &boxes[j * 5]) > thresh)
                is_dead[j] = 1;
    }
    num_keep = count;
}

}  // namespace rcnn

}  // namespace dragon

}  // namespace ts
