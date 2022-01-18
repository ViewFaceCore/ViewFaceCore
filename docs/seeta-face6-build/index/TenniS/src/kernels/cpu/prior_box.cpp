//
// Created by kier on 19-6-17.
//

#include "runtime/operator.h"
#include "runtime/stack.h"
#include "global/operator_factory.h"
#include "core/tensor_builder.h"
#include <cmath>
#include <algorithm>
namespace ts {
    namespace cpu {
        class PriorBox : public Operator {
        public:
            std::vector<float> min_sizes_;
            std::vector<float> max_sizes_;
            std::vector<float> aspect_ratios_;
            bool flip_ = true;
            int num_priors_ = 0;
            bool clip_ = false;
            std::vector<float> variance_;

            int img_w_ = 0;
            int img_h_ = 0;
            float step_w_ = 0;
            float step_h_ = 0;

            float offset_ = 0.5;

            PriorBox() {
                field("min_size", REQUIRED);
                field("max_size", REQUIRED);
                field("aspect_ratio", REQUIRED);
                field("flip", OPTIONAL, tensor::from<bool>(true));
                field("clip", OPTIONAL, tensor::from<bool>(false));
                field("variance", OPTIONAL, tensor::from<float>(0.1f));
                field("offset", OPTIONAL, tensor::from<float>(0.5f));
                field("img_w", OPTIONAL);
                field("img_h", OPTIONAL);
                field("img_size", OPTIONAL);
                field("step_w", OPTIONAL);
                field("step_h", OPTIONAL);
                field("step", OPTIONAL);
            }

            void init() final {
                min_sizes_ = tensor::array::to_float(get("min_size"));
                TS_AUTO_ASSERT(!min_sizes_.empty());
                for (auto &size : min_sizes_) {
                    TS_CHECK_GT(size, 0) << "min_size must be positive." << eject;
                }

                aspect_ratios_.clear();
                aspect_ratios_.push_back(1.);
                flip_ = tensor::to_bool(get("flip"));
                auto tmp_aspect_ratio = tensor::array::to_float(get("aspect_ratio"));
                for (size_t i = 0; i < tmp_aspect_ratio.size(); ++i) {
                    float ar = tmp_aspect_ratio[i];
                    bool already_exist = false;
                    for (size_t j = 0; j < aspect_ratios_.size(); ++j) {
                        if (std::fabs(ar - aspect_ratios_[j]) < 1e-6) {
                            already_exist = true;
                            break;
                        }
                    }
                    if (!already_exist) {
                        aspect_ratios_.push_back(ar);
                        if (flip_) {
                            aspect_ratios_.push_back(1. / ar);
                        }
                    }
                }
                num_priors_ = int(aspect_ratios_.size() * min_sizes_.size());
                max_sizes_ = tensor::array::to_float(get("max_size"));

                if (max_sizes_.size() > 0) {
                    TS_AUTO_CHECK_EQ(max_sizes_.size(), min_sizes_.size());
                    for (size_t i = 0; i < max_sizes_.size(); ++i) {
                        TS_AUTO_CHECK_GT(max_sizes_[i], min_sizes_[i])
                                << "max_size must be greater than min_size.";
                        num_priors_ += 1;
                    }
                }
                clip_ = tensor::to_bool(get("clip"));
                auto tmp_variance = tensor::array::to_float(get("variance"));
                if (tmp_variance.size() > 1) {
                    // Must and only provide 4 variance.
                    TS_AUTO_CHECK_EQ(tmp_variance.size(), 4);
                    for (size_t i = 0; i < tmp_variance.size(); ++i) {
                        TS_AUTO_CHECK_GT(tmp_variance[i], 0);
                        variance_.push_back(tmp_variance[i]);
                    }
                } else if (tmp_variance.size() == 1) {
                    TS_AUTO_CHECK_GT(tmp_variance[0], 0);
                    variance_.push_back(tmp_variance[0]);
                } else {
                    // Set default to 0.1.
                    variance_.push_back(0.1);
                }

                if (has("img_h") || has("img_w")) {
                    TS_CHECK(!has("img_size"))
                            << "Either img_size or img_h/img_w should be specified; not both." << eject;
                    img_h_ = tensor::to_int(get("img_h"));
                    TS_CHECK_GT(img_h_, 0) << "img_h should be larger than 0." << eject;
                    img_w_ = tensor::to_int(get("img_w"));
                    TS_CHECK_GT(img_w_, 0) << "img_w should be larger than 0." << eject;
                } else if (has("img_size")) {
                    const int img_size = tensor::to_int(get("img_size"));
                    TS_CHECK_GT(img_size, 0) << "img_size should be larger than 0." << eject;
                    img_h_ = img_size;
                    img_w_ = img_size;
                } else {
                    img_h_ = 0;
                    img_w_ = 0;
                }

                if (has("step_h") || has("step_w")) {
                    TS_CHECK(!has("step"))
                            << "Either step or step_h/step_w should be specified; not both." << eject;
                    step_h_ = tensor::to_int(get("step_h"));
                    TS_CHECK_GT(step_h_, 0.) << "step_h should be larger than 0." << eject;
                    step_w_ = tensor::to_int(get("step_w"));
                    TS_CHECK_GT(step_w_, 0.) << "step_w should be larger than 0." << eject;
                } else if (has("step")) {
                    const float step = tensor::to_float(get("step"));
                    TS_CHECK_GT(step, 0) << "step should be larger than 0." << eject;
                    step_h_ = step;
                    step_w_ = step;
                } else {
                    step_h_ = 0;
                    step_w_ = 0;
                }

                offset_ = tensor::to_float(get("offset"));
            }

            std::vector<int> infer_top_shape(const Tensor &feat) {
                const int layer_width = feat.size(3);
                const int layer_height = feat.size(2);

                std::vector<int> top_shape(3, 1);
                top_shape[0] = 1;
                top_shape[1] = 2;
                top_shape[2] = layer_width * layer_height * num_priors_ * 4;

                if (top_shape[2] < 0) top_shape[2] = -1;    // -1  for ?

                return top_shape;
            }

            int infer(Stack &stack, std::vector<Tensor::Prototype> &output) final {
                TS_AUTO_ASSERT(stack.size() >= 1);

                output.resize(1);
                output[0] = Tensor::Prototype(stack[0].dtype(), infer_top_shape(stack[0]));
                return 1;
            }

            static int Offset(const Tensor &x, const int n, const int c = 0, const int h = 0) {
                auto &size = x.sizes();
                return (n * size[1] + c) * size[2] + h;
            }

            int run(Stack &stack) final {
                TS_AUTO_CHECK(stack.size() >= 1);
                auto feat = tensor::cast(FLOAT32, stack[0]);
                auto &top = *stack.push(FLOAT32, infer_top_shape(feat), CPU);
                using Dtype = float;

                const int layer_width = feat.size(3);
                const int layer_height = feat.size(2);
                int img_width, img_height;
                if (img_h_ == 0 || img_w_ == 0) {
                    TS_AUTO_ASSERT(stack.size() > 1);
                    auto &img = stack[1];
                    img_width = img.size(3);
                    img_height = img.size(2);
                } else {
                    img_width = img_w_;
                    img_height = img_h_;
                }
                float step_w, step_h;
                if (step_w_ == 0 || step_h_ == 0) {
                    step_w = static_cast<float>(img_width) / layer_width;
                    step_h = static_cast<float>(img_height) / layer_height;
                } else {
                    step_w = step_w_;
                    step_h = step_h_;
                }
                auto *top_data = top.data<Dtype>();
                int dim = layer_height * layer_width * num_priors_ * 4;
                int idx = 0;
                for (int h = 0; h < layer_height; ++h) {
                    for (int w = 0; w < layer_width; ++w) {
                        float center_x = (w + offset_) * step_w;
                        float center_y = (h + offset_) * step_h;
                        float box_width, box_height;
                        for (int s = 0; s < min_sizes_.size(); ++s) {
                            int min_size_ = min_sizes_[s];
                            // first prior: aspect_ratio = 1, size = min_size
                            box_width = box_height = min_size_;
                            // xmin
                            top_data[idx++] = (center_x - box_width / 2.) / img_width;
                            // ymin
                            top_data[idx++] = (center_y - box_height / 2.) / img_height;
                            // xmax
                            top_data[idx++] = (center_x + box_width / 2.) / img_width;
                            // ymax
                            top_data[idx++] = (center_y + box_height / 2.) / img_height;

                            if (max_sizes_.size() > 0) {
                                TS_AUTO_CHECK_EQ(min_sizes_.size(), max_sizes_.size());
                                int max_size_ = max_sizes_[s];
                                // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                                box_width = box_height = std::sqrt(min_size_ * max_size_);
                                // xmin
                                top_data[idx++] = (center_x - box_width / 2.) / img_width;
                                // ymin
                                top_data[idx++] = (center_y - box_height / 2.) / img_height;
                                // xmax
                                top_data[idx++] = (center_x + box_width / 2.) / img_width;
                                // ymax
                                top_data[idx++] = (center_y + box_height / 2.) / img_height;
                            }

                            // rest of priors
                            for (size_t r = 0; r < aspect_ratios_.size(); ++r) {
                                float ar = aspect_ratios_[r];
                                if (fabs(ar - 1.) < 1e-6) {
                                    continue;
                                }
                                box_width = min_size_ * std::sqrt(ar);
                                box_height = min_size_ / std::sqrt(ar);
                                // xmin
                                top_data[idx++] = (center_x - box_width / 2.) / img_width;
                                // ymin
                                top_data[idx++] = (center_y - box_height / 2.) / img_height;
                                // xmax
                                top_data[idx++] = (center_x + box_width / 2.) / img_width;
                                // ymax
                                top_data[idx++] = (center_y + box_height / 2.) / img_height;
                            }
                        }
                    }
                }
                // clip the prior's coordidate such that it is within [0, 1]
                if (clip_) {
                    for (int d = 0; d < dim; ++d) {
                        top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
                    }
                }
                // set the variance.
                top_data +=  Offset(top, 0, 1);
                if (variance_.size() == 1) {
                    auto src = Dtype(variance_[0]);
                    memset(top_data, CPU, dim * sizeof(Dtype),
                            &src, CPU, sizeof(Dtype));
                } else {
                    int count = 0;
                    for (int h = 0; h < layer_height; ++h) {
                        for (int w = 0; w < layer_width; ++w) {
                            for (int i = 0; i < num_priors_; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    top_data[count] = variance_[j];
                                    ++count;
                                }
                            }
                        }
                    }
                }

                return 1;
            }
        };
    }
}

using namespace ts;
using namespace cpu;
TS_REGISTER_OPERATOR(PriorBox, CPU, "prior_box")