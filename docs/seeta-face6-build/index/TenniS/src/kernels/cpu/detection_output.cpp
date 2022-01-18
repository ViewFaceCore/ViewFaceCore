//
// Created by kier on 19-6-17.
//

#include "runtime/operator.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"

#include "kernels/cpu/caffe/bbox_util.hpp"
#include "utils/need.h"

#include <functional>
#include <algorithm>

#include "kernels/cpu/caffe/blob.hpp"

namespace ts {
    namespace cpu {
        using namespace caffe;

        template <typename T>
        void caffe_set(const int N, const T alpha, T *X) {
            memset(X, CPU, N * sizeof(T),
                   &alpha, CPU, sizeof(T));
        }

        class DetectionOutput : public Operator {
        public:
            using self = DetectionOutput;

            int num_classes_ = 0;
            bool share_location_ = true;
            int num_loc_classes_ = 0;
            int background_label_id_ = 0;
            CodeType code_type_ = CodeType::CORNER;
            bool variance_encoded_in_target_ = true;
            int keep_top_k_ = 100;
            float confidence_threshold_ = 0.01;

            int num_ = 0;
            int num_priors_ = 0;

            float nms_threshold_ = 0.3;
            int top_k_ = 1000;
            float eta_ = 1.0;

//            Tensor bbox_preds_;
//            Tensor bbox_permute_;
//            Tensor conf_permute_;

            DetectionOutput() {
                field("num_classes", REQUIRED);
                field("share_location", OPTIONAL, tensor::from<bool>(true));
                field("background_label_id", OPTIONAL, tensor::from<int32_t>(0));
                field("nms_threshold", OPTIONAL, tensor::from<float>(0.3f));
                field("nms_top_k", OPTIONAL, tensor::from<int>(1000));
                field("nms_eta", OPTIONAL, tensor::from<float>(1.0f));
                
                field("code_type", OPTIONAL, tensor::from<int>(1));
                field("variance_encoded_in_target", OPTIONAL, tensor::from<bool>(false));
                field("keep_top_k", OPTIONAL, tensor::from<int>(-1));
                field("confidence_threshold", OPTIONAL, tensor::from<float>(1e-5f));
            }

            void init() final {
                num_classes_ = tensor::to_int(get("num_classes"));
                share_location_ = tensor::to_bool(get("share_location"));
                num_loc_classes_ = share_location_ ? 1 : num_classes_;
                background_label_id_ = tensor::to_int(get("background_label_id"));
                code_type_ = CodeType(tensor::to_int(get("code_type")));
                variance_encoded_in_target_ =
                        tensor::to_bool(get("variance_encoded_in_target"));
                keep_top_k_ = tensor::to_int(get("keep_top_k"));
                confidence_threshold_ = has("confidence_threshold") ?
                                        tensor::to_float(get("confidence_threshold")) : -FLT_MAX;
                // Parameters used in nms.
                nms_threshold_ = tensor::to_float(get("nms_threshold"));
                TS_CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative." << eject;
                eta_ = tensor::to_float(get("nms_eta"));
                TS_AUTO_CHECK_GT(eta_, 0.);
                TS_AUTO_CHECK_LE(eta_, 1.);
                top_k_ = -1;
                if (has("nms_top_k")) {
                    top_k_ = tensor::to_int(get("nms_top_k"));
                }
            }

            std::vector<int> infer_top_shape(Stack &stack, bool alloc_buffer = false) {
                auto &bottom = stack;
                TS_AUTO_CHECK_EQ(bottom[0].size(0), bottom[1].size(0));

                // malloc memory, remeber to free then
                if (alloc_buffer) {
//                    bbox_preds_ = stack.make(bottom[0].proto(), CPU);
//                    bbox_permute_ = stack.make(bottom[0].proto(), CPU);
//                    conf_permute_ = stack.make(bottom[1].proto(), CPU);
                }

                num_priors_ = bottom[2].size(2) / 4;
                TS_CHECK_EQ(num_priors_ * num_loc_classes_ * 4, bottom[0].size(1))
                        << "Number of priors must match number of location predictions." << eject;
                TS_CHECK_EQ(num_priors_ * num_classes_, bottom[1].size(1))
                        << "Number of priors must match number of confidence predictions." << eject;
                // num() and channels() are 1.
                std::vector<int> top_shape(2, 1);
                // Since the number of bboxes to be kept is unknown before nms, we manually
                // set it to (fake) 1.
                top_shape.push_back(1);
                // Each row is a 7 dimension vector, which stores
                // [image_id, label, confidence, xmin, ymin, xmax, ymax]
                top_shape.push_back(7);

                return top_shape;
            }

            void clear_buffer() {
//                bbox_preds_ = Tensor();
//                bbox_permute_ = Tensor();
//                conf_permute_ = Tensor();
            }

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) final {
                TS_AUTO_ASSERT(stack.size() >= 3 && stack.size() <= 4);

                output.resize(1);
                output[0] = Tensor::Prototype(stack[0].dtype(), infer_top_shape(stack, false));
                return 1;
            }

            static int Offset(const Tensor &x, const int n, const int c = 0, const int h = 0,
                       const int w = 0) {
                auto &size = x.sizes();
                return ((n * size[1] + c) * size[2] + h) * size[3] + w;
            }

            int run(Stack &stack) final {
                TS_AUTO_ASSERT(stack.size() >= 3 && stack.size() <= 4);
                using Dtype = float;
                auto loc = tensor::cast(FLOAT32,  stack[0]);
                auto conf = tensor::cast(FLOAT32,  stack[1]);
                auto prior = tensor::cast(FLOAT32,  stack[2]);
                auto &bottom = stack;
                std::vector<Blob<float>> top(1);
                top[0].Reshape(infer_top_shape(stack, true));

                auto loc_data = loc.data<Dtype>();
                auto conf_data = conf.data<Dtype>();
                auto prior_data = prior.data<Dtype>();
                const int num = bottom[0].size(0);

                // Retrieve all location predictions.
                std::vector<LabelBBox> all_loc_preds;
                GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
                                  share_location_, &all_loc_preds);

                // Retrieve all confidences.
                vector<map<int, vector<float> > > all_conf_scores;
                GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
                                    &all_conf_scores);

                // Retrieve all prior bboxes. It is same within a batch since we assume all
                // images in a batch are of same dimension.
                vector<NormalizedBBox> prior_bboxes;
                vector<vector<float> > prior_variances;
                GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

                // Decode all loc predictions to bboxes.
                vector<LabelBBox> all_decode_bboxes;
                const bool clip_bbox = false;
                DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
                                share_location_, num_loc_classes_, background_label_id_,
                                code_type_, variance_encoded_in_target_, clip_bbox,
                                &all_decode_bboxes);

                int num_kept = 0;
                vector<map<int, vector<int> > > all_indices;
                for (int i = 0; i < num; ++i) {
                    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
                    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
                    map<int, vector<int> > indices;
                    int num_det = 0;
                    for (int c = 0; c < num_classes_; ++c) {
                        if (c == background_label_id_) {
                            // Ignore background class.
                            continue;
                        }
                        if (conf_scores.find(c) == conf_scores.end()) {
                            // Something bad happened if there are no predictions for current label.
                            TS_LOG(LOG_FATAL) << "Could not find confidence predictions for label " << c << eject;
                        }
                        const vector<float>& scores = conf_scores.find(c)->second;
                        int label = share_location_ ? -1 : c;
                        if (decode_bboxes.find(label) == decode_bboxes.end()) {
                            // Something bad happened if there are no predictions for current label.
                            TS_LOG(LOG_FATAL) << "Could not find location predictions for label " << label << eject;
                            continue;
                        }
                        const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
                        ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
                                     top_k_, &(indices[c]));
                        num_det += indices[c].size();
                    }
                    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
                        vector<pair<float, pair<int, int> > > score_index_pairs;
                        for (map<int, vector<int> >::iterator it = indices.begin();
                                it != indices.end(); ++it) {
                            int label = it->first;
                            const vector<int>& label_indices = it->second;
                            if (conf_scores.find(label) == conf_scores.end()) {
                                // Something bad happened for current label.
                                TS_LOG(LOG_FATAL) << "Could not find location predictions for " << label << eject;
                                continue;
                            }
                            const vector<float>& scores = conf_scores.find(label)->second;
                            for (int j = 0; j < label_indices.size(); ++j) {
                                int idx = label_indices[j];
                                TS_AUTO_CHECK_LT(idx, scores.size());
                                score_index_pairs.push_back(std::make_pair(
                                        scores[idx], std::make_pair(label, idx)));
                            }
                        }
                        // Keep top k results per image.
                        std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                                  SortScorePairDescend<pair<int, int> >);
                        score_index_pairs.resize(keep_top_k_);
                        // Store the new indices.
                        map<int, vector<int> > new_indices;
                        for (int j = 0; j < score_index_pairs.size(); ++j) {
                            int label = score_index_pairs[j].second.first;
                            int idx = score_index_pairs[j].second.second;
                            new_indices[label].push_back(idx);
                        }
                        all_indices.push_back(new_indices);
                        num_kept += keep_top_k_;
                    } else {
                        all_indices.push_back(indices);
                        num_kept += num_det;
                    }
                }

                vector<int> top_shape(2, 1);
                top_shape.push_back(num_kept);
                top_shape.push_back(7);
                Dtype* top_data;
                if (num_kept == 0) {
                    // TS_LOG(LOG_INFO) << "Couldn't find any detections";
                    top_shape[2] = num;
                    top[0]->Reshape(top_shape);
                    top_data = top[0]->mutable_cpu_data();
                    caffe_set<Dtype>(top[0]->count(), -1, top_data);
                    // Generate fake results per image.
                    for (int i = 0; i < num; ++i) {
                        top_data[0] = i;
                        top_data += 7;
                    }
                } else {
                    top[0]->Reshape(top_shape);
                    top_data = top[0]->mutable_cpu_data();
                }

                int count = 0;
                for (int i = 0; i < num; ++i) {
                    const map<int, vector<float> >& conf_scores = all_conf_scores[i];
                    const LabelBBox& decode_bboxes = all_decode_bboxes[i];
                    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
                            it != all_indices[i].end(); ++it) {
                        int label = it->first;
                        if (conf_scores.find(label) == conf_scores.end()) {
                            // Something bad happened if there are no predictions for current label.
                            TS_LOG(LOG_FATAL) << "Could not find confidence predictions for " << label << eject;
                            continue;
                        }
                        const vector<float>& scores = conf_scores.find(label)->second;
                        int loc_label = share_location_ ? -1 : label;
                        if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                            // Something bad happened if there are no predictions for current label.
                            TS_LOG(LOG_FATAL) << "Could not find location predictions for " << loc_label << eject;
                            continue;
                        }
                        const vector<NormalizedBBox>& bboxes =
                                decode_bboxes.find(loc_label)->second;
                        vector<int>& indices = it->second;
                        for (int j = 0; j < indices.size(); ++j) {
                            int idx = indices[j];
                            top_data[count * 7] = i;
                            top_data[count * 7 + 1] = label;
                            top_data[count * 7 + 2] = scores[idx];
                            const NormalizedBBox& bbox = bboxes[idx];
                            top_data[count * 7 + 3] = bbox.xmin();
                            top_data[count * 7 + 4] = bbox.ymin();
                            top_data[count * 7 + 5] = bbox.xmax();
                            top_data[count * 7 + 6] = bbox.ymax();

                            ++count;
                        }
                    }
                }

                stack.push(top[0].tensor());

                return 1;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(DetectionOutput, CPU, "detection_output")