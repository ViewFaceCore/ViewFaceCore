#include "kernels/cpu/non_max_suppression_v3.h"
#include "global/operator_factory.h"
#include "backend/name.h"
#include <vector>
#include <numeric>
#include <queue>

namespace ts {
    namespace cpu {

        template <typename T>
        static inline bool check_suppress(
                            const T* boxes, int i, int j, T iou_threshold, int mode) {

            T ymin_i,xmin_i,ymax_i,xmax_i;
            T ymin_j,xmin_j,ymax_j,xmax_j;
            T area_i,area_j;

            if(mode == 1) {
                ymin_i = std::min<T>(boxes[i * 4 + 0], boxes[i * 4 + 0] + boxes[i * 4 + 2]);
                xmin_i = std::min<T>(boxes[i * 4 + 1], boxes[i * 4 + 1] + boxes[i * 4 + 3]);
                ymax_i = std::max<T>(boxes[i * 4 + 0], boxes[i * 4 + 0] + boxes[i * 4 + 2]);
                xmax_i = std::max<T>(boxes[i * 4 + 1], boxes[i * 4 + 1] + boxes[i * 4 + 3]);

                ymin_j = std::min<T>(boxes[j * 4 + 0], boxes[j * 4 + 0] + boxes[j * 4 + 2]);
                xmin_j = std::min<T>(boxes[j * 4 + 1], boxes[j * 4 + 1] + boxes[j * 4 + 3]);
                ymax_j = std::max<T>(boxes[j * 4 + 0], boxes[j * 4 + 0] + boxes[j * 4 + 2]);
                xmax_j = std::max<T>(boxes[j * 4 + 1], boxes[j * 4 + 1] + boxes[j * 4 + 3]);

            }else {
                ymin_i = std::min<T>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
                xmin_i = std::min<T>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
                ymax_i = std::max<T>(boxes[i * 4 + 0], boxes[i * 4 + 2]);
                xmax_i = std::max<T>(boxes[i * 4 + 1], boxes[i * 4 + 3]);
                ymin_j = std::min<T>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
                xmin_j = std::min<T>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
                ymax_j = std::max<T>(boxes[j * 4 + 0], boxes[j * 4 + 2]);
                xmax_j = std::max<T>(boxes[j * 4 + 1], boxes[j * 4 + 3]);
            }

            area_i = (ymax_i - ymin_i  ) * (xmax_i - xmin_i );
            area_j = (ymax_j - ymin_j  ) * (xmax_j - xmin_j );

            if (area_i <= static_cast<T>(0) || area_j <= static_cast<T>(0)) return 0;
            const T intersection_ymin = std::max<T>(ymin_i, ymin_j);
            const T intersection_xmin = std::max<T>(xmin_i, xmin_j);
            const T intersection_ymax = std::min<T>(ymax_i, ymax_j);
            const T intersection_xmax = std::min<T>(xmax_i, xmax_j);

            const T intersection_area =
            std::max<T>(intersection_ymax - intersection_ymin  , static_cast<T>(0.0)) *
            std::max<T>(intersection_xmax - intersection_xmin  , static_cast<T>(0.0));
            const T iou = intersection_area / (area_i + area_j - intersection_area);

            return iou > iou_threshold;
       }


        template <typename T>
        static void cpu_non_max_suppression_v3_compute_run(const Tensor &x, const Tensor &scores,
                             int max_output, float iou_threshold, float score_threshold,
                             const std::string & mode, Tensor &out) {
            auto *p_outdata = out.data<int32_t>();
            const T *p_xdata  = x.data<T>();

            auto scores_data = scores.data<float>();

            int nmode = 1;  //xywh
            if(mode == "xyxy") {
                nmode = 0;
            }

            struct Candidate {
                int box_index;
                float score;
            };

            auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
                return bs_i.score < bs_j.score;
            };
            std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
                 candidate_priority_queue(cmp);
            for (int i = 0; i < scores.count(); ++i) {
                if (scores_data[i] > score_threshold) {
                    candidate_priority_queue.emplace(Candidate({i, scores_data[i]}));
                }
            }

            std::vector<int> selected;
            std::vector<float> selected_scores;
            Candidate next_candidate;
 
            while (selected.size() < max_output && !candidate_priority_queue.empty()) {
                next_candidate = candidate_priority_queue.top();
                candidate_priority_queue.pop();

                bool should_select = true;

                for (int j = static_cast<int>(selected.size()) - 1; j >= 0; --j) {
                    if (check_suppress<T>(p_xdata, next_candidate.box_index, selected[j], T(iou_threshold), nmode)) {
                        should_select = false;
                        break;
                    }
                }

                if (should_select) {
                    selected.push_back(next_candidate.box_index);
                    selected_scores.push_back(next_candidate.score);
                }
            }

            ::memset(p_outdata, -1, sizeof(int32_t) * max_output);
            for(int i=0; i<selected.size(); i++) {
                p_outdata[i] = selected[i];
            }

        }

        void Non_Max_Suppression_V3::non_max_suppression_v3(const Tensor &x, const Tensor & scores, Tensor &out) {
            DTYPE dtype = x.dtype();
            switch (dtype) {
#define DECLARE_COMPUTE_RUN(DTYPE, TYPE) \
        case DTYPE: { cpu_non_max_suppression_v3_compute_run<TYPE>(x, scores,m_max_output_size, m_iou_threshold,m_score_threshold,m_mode, out); break; }
                //DECLARE_COMPUTE_RUN(INT8, int8_t);
                //DECLARE_COMPUTE_RUN(UINT8, uint8_t);
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
TS_REGISTER_OPERATOR(Non_Max_Suppression_V3, CPU, name::layer::non_max_suppression_v3())
