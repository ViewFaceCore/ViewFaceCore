#include "kernels/cpu/dragon/bbox_utils.h"
#include "core/ieee754_float.h"

#include "kernels/common/third/dragon.h"

#ifdef TS_USE_CUDA_FP16
#include "kernels/gpu/cudax_fp16_math.h"
#endif
#include "kernels/gpu/gpu_kernel.h"

namespace ts {
    namespace dragon {
        namespace rcnn {
        /******************** BBox ********************/

            template<typename T>
            __device__ int _BBoxTransform(
                    const T dx,
                    const T dy,
                    const T d_log_w,
                    const T d_log_h,
                    const T im_w,
                    const T im_h,
                    const T min_box_w,
                    const T min_box_h,
                    T *bbox) {
                const T w = bbox[2] - bbox[0] + (T) 1;
                const T h = bbox[3] - bbox[1] + (T) 1;
                const T ctr_x = bbox[0] + (T) 0.5 * w;
                const T ctr_y = bbox[1] + (T) 0.5 * h;

                const T pred_ctr_x = dx * w + ctr_x;
                const T pred_ctr_y = dy * h + ctr_y;
                const T pred_w = exp(d_log_w) * w;
                const T pred_h = exp(d_log_h) * h;

                bbox[0] = pred_ctr_x - (T) 0.5 * pred_w;
                bbox[1] = pred_ctr_y - (T) 0.5 * pred_h;
                bbox[2] = pred_ctr_x + (T) 0.5 * pred_w;
                bbox[3] = pred_ctr_y + (T) 0.5 * pred_h;

                bbox[0] = max((T) 0, min(bbox[0], im_w - (T) 1));
                bbox[1] = max((T) 0, min(bbox[1], im_h - (T) 1));
                bbox[2] = max((T) 0, min(bbox[2], im_w - (T) 1));
                bbox[3] = max((T) 0, min(bbox[3], im_h - (T) 1));

                const T box_w = bbox[2] - bbox[0] + (T) 1;
                const T box_h = bbox[3] - bbox[1] + (T) 1;
                return (box_w >= min_box_w) * (box_h >= min_box_h);
            }

            /******************** Proposal ********************/
            template<typename T>
            __global__ void _GenerateProposals(
                    const int nthreads,
                    const int A,
                    const int feat_h,
                    const int feat_w,
                    const int stride,
                    const float im_h,
                    const float im_w,
                    const float min_box_h,
                    const float min_box_w,
                    const T *scores,
                    const T *bbox_deltas,
                    const T *anchors,
                    T *proposals) {
                CUDA_1D_KERNEL_LOOP(idx, nthreads) {
                    const int h = idx / A / feat_w;
                    const int w = (idx / A) % feat_w;
                    const int a = idx % A;
                    const T x = w * stride;
                    const T y = h * stride;
                    const T *bbox_delta = bbox_deltas + h * feat_w + w;
                    const T *score = scores + h * feat_w + w;
                    const int K = feat_h * feat_w;
                    const T dx = bbox_delta[(a * 4 + 0) * K];
                    const T dy = bbox_delta[(a * 4 + 1) * K];
                    const T d_log_w = bbox_delta[(a * 4 + 2) * K];
                    const T d_log_h = bbox_delta[(a * 4 + 3) * K];
                    T *proposal = proposals + idx * 5;
                    proposal[0] = x + anchors[a * 4 + 0];
                    proposal[1] = y + anchors[a * 4 + 1];
                    proposal[2] = x + anchors[a * 4 + 2];
                    proposal[3] = y + anchors[a * 4 + 3];
                    proposal[4] = _BBoxTransform(
                            dx, dy, d_log_w, d_log_h,
                            im_w, im_h, min_box_w, min_box_h,
                            proposal) * score[a * K];
                }
            }

            template<>
            void GenerateProposals<float, CUDAContext>(
                    const int A,
                    const int feat_h,
                    const int feat_w,
                    const int stride,
                    const float im_h,
                    const float im_w,
                    const float min_box_h,
                    const float min_box_w,
                    const float *scores,
                    const float *bbox_deltas,
                    const float *anchors,
                    float *proposals,
                    CUDAContext *ctx) {
                const auto num_proposals = A * feat_h * feat_w;
                RUN_KERNEL_STREAM(_GenerateProposals<float>,
                                  CUDA_BLOCKS(num_proposals), CUDA_THREADS,
                                  0, ctx->cuda_stream(),
                                  num_proposals, A, feat_h, feat_w, stride,
                                  im_h, im_w, min_box_h, min_box_w,
                                  scores, bbox_deltas, anchors, proposals);
            }

            template<typename T>
            __global__ void _GenerateProposals_v2(
                    const int nthreads,
                    const float im_h,
                    const float im_w,
                    const float min_box_h,
                    const float min_box_w,
                    const T *scores,
                    const T *bbox_deltas,
                    T *proposals) {
                CUDA_1D_KERNEL_LOOP(idx, nthreads) {
                    const float dx = bbox_deltas[idx];
                    const float dy = bbox_deltas[nthreads + idx];
                    const float d_log_w = bbox_deltas[2 * nthreads + idx];
                    const float d_log_h = bbox_deltas[3 * nthreads + idx];
                    T *proposal = proposals + idx * 5;
                    proposal[4] = _BBoxTransform(
                            dx, dy, d_log_w, d_log_h,
                            im_w, im_h, min_box_w, min_box_h,
                            proposal) * scores[idx];
                }
            }

            template<>
            void GenerateProposals_v2<float, CUDAContext>(
                    const int total_anchors,
                    const float im_h,
                    const float im_w,
                    const float min_box_h,
                    const float min_box_w,
                    const float *scores,
                    const float *bbox_deltas,
                    float *proposals,
                    CUDAContext *ctx) {
                RUN_KERNEL_STREAM(_GenerateProposals_v2<float>,
                                  CUDA_BLOCKS(total_anchors), CUDA_THREADS,
                                  0, ctx->cuda_stream(),
                                  total_anchors, im_h, im_w, min_box_h, min_box_w,
                                  scores, bbox_deltas, proposals);
            }

            /******************** NMS ********************/

#define DIV_UP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define NMS_BLOCK_SIZE 64

            template<typename T>
            __device__  T iou(const T *A, const T *B) {
                const T x1 = max(A[0], B[0]);
                const T y1 = max(A[1], B[1]);
                const T x2 = min(A[2], B[2]);
                const T y2 = min(A[3], B[3]);
                const T width = max((T) 0, x2 - x1 + 1);
                const T height = max((T) 0, y2 - y1 + 1);
                const T area = width * height;
                const T A_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
                const T B_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
                return area / (A_area + B_area - area);
            }

            template<typename T>
            __global__ void nms_mask(
                    const int num_boxes,
                    const T nms_thresh,
                    const T *boxes,
                    uint64_t *mask) {
                const int i_start = blockIdx.x * NMS_BLOCK_SIZE;
                const int di_end = min(num_boxes - i_start, NMS_BLOCK_SIZE);
                const int j_start = blockIdx.y * NMS_BLOCK_SIZE;
                const int dj_end = min(num_boxes - j_start, NMS_BLOCK_SIZE);

                const int num_blocks = DIV_UP(num_boxes, NMS_BLOCK_SIZE);
                const int bid = blockIdx.x;
                const int tid = threadIdx.x;

                __shared__ T boxes_i[NMS_BLOCK_SIZE * 4];

                if (tid < di_end) {
                    boxes_i[tid * 4 + 0] = boxes[(i_start + tid) * 5 + 0];
                    boxes_i[tid * 4 + 1] = boxes[(i_start + tid) * 5 + 1];
                    boxes_i[tid * 4 + 2] = boxes[(i_start + tid) * 5 + 2];
                    boxes_i[tid * 4 + 3] = boxes[(i_start + tid) * 5 + 3];
                }

                __syncthreads();

                if (tid < dj_end) {
                    const T *const box_j = boxes + (j_start + tid) * 5;
                    unsigned long long mask_j = 0;
                    const int di_start = (i_start == j_start) ? (tid + 1) : 0;
                    for (int di = di_start; di < di_end; ++di)
                        if (iou(box_j, boxes_i + di * 4) > nms_thresh)
                            mask_j |= 1ULL << di;
                    mask[(j_start + tid) * num_blocks + bid] = mask_j;
                }
            }

            template<typename T>
            void _ApplyNMS(
                    const int num_boxes,
                    const int max_keeps,
                    const float thresh,
                    const T *boxes,
                    int *keep_indices,
                    int &num_keep,
                    CUDAContext *ctx) {
                const int num_blocks = DIV_UP(num_boxes, NMS_BLOCK_SIZE);
                const dim3 blocks(num_blocks, num_blocks);
                size_t mask_nbytes = num_boxes * num_blocks * sizeof(uint64_t);
                size_t boxes_nbytes = num_boxes * 5 * sizeof(T);

                void *boxes_dev, *mask_dev;
                CUDA_CHECK(cudaMalloc(&boxes_dev, boxes_nbytes));
                CUDA_CHECK(cudaMalloc(&mask_dev, mask_nbytes));
                CUDA_CHECK(cudaMemcpy(boxes_dev, boxes,
                                      boxes_nbytes, cudaMemcpyHostToDevice));
                RUN_KERNEL_STREAM(nms_mask<T>,
                                  blocks, NMS_BLOCK_SIZE,
                                  0, ctx->cuda_stream(), num_boxes,
                                  thresh, (T *) boxes_dev, (uint64_t *) mask_dev);
                CUDA_CHECK(cudaPeekAtLastError());

                std::vector<uint64_t> mask_host(num_boxes * num_blocks);
                CUDA_CHECK(cudaMemcpy(&mask_host[0], mask_dev,
                                      mask_nbytes, cudaMemcpyDeviceToHost));

                std::vector<uint64_t> dead_bit(num_blocks);
                std::memset(&dead_bit[0], 0, sizeof(uint64_t) * num_blocks);
                int num_selected = 0;

                for (int i = 0; i < num_boxes; ++i) {
                    const int nblock = i / NMS_BLOCK_SIZE;
                    const int inblock = i % NMS_BLOCK_SIZE;
                    if (!(dead_bit[nblock] & (1ULL << inblock))) {
                        keep_indices[num_selected++] = i;
                        uint64_t *mask_i = &mask_host[0] + i * num_blocks;
                        for (int j = nblock; j < num_blocks; ++j) dead_bit[j] |= mask_i[j];
                        if (num_selected == max_keeps) break;
                    }
                }
                num_keep = num_selected;
                CUDA_CHECK(cudaFree(mask_dev));
                CUDA_CHECK(cudaFree(boxes_dev));
            }

            template<>
            void ApplyNMS<float, CUDAContext>(
                    const int num_boxes,
                    const int max_keeps,
                    const float thresh,
                    const float *boxes,
                    int *keep_indices,
                    int &num_keep,
                    CUDAContext *ctx) {
                _ApplyNMS<float>(num_boxes, max_keeps, thresh,
                                 boxes, keep_indices, num_keep, ctx);
            }

        }  // namespace rcnn

    }  // namespace dragon

}  // namespace ts
