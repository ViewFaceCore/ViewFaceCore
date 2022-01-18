//
// Created by kier on 19-7-29.
//

#include "backend/zoo/crop_nd.h"
#include "backend/zoo/divided.h"
#include "backend/zoo/limit.h"
#include "backend/zoo/nhwc_center_crop2d.h"
#include "backend/zoo/nhwc_letterbox.h"
#include "backend/zoo/nhwc_scale_resize2d.h"

#include "kernels/cpu/conv2d_v2.h"
#include "kernels/cpu/pooling2d_v2.h"
#include "kernels/cpu/depthwise_conv2d_v2.h"
#include "kernels/cpu/concat.h"
#include "kernels/cpu/gather.h"
#include "kernels/cpu/gatherv2.h"
#include "kernels/cpu/stack_tensor.h"
#include "kernels/cpu/strided_slice.h"

#include "global/operator_factory.h"
#include "backend/name.h"

using namespace ts;

using namespace zoo;
TS_REGISTER_OPERATOR(CropND, GPU, name::layer::crop_nd())
TS_REGISTER_OPERATOR(Divided, GPU, name::layer::divided())
TS_REGISTER_OPERATOR(Limit, GPU, name::layer::limit())
TS_REGISTER_OPERATOR(NHWCCenterCrop2D, GPU, name::layer::nhwc_center_crop2d())
TS_REGISTER_OPERATOR(NHWCLetterBox, GPU, name::layer::nhwc_letterbox())
TS_REGISTER_OPERATOR(NHWCScaleResize2D, GPU, name::layer::nhwc_scale_resize2d())

using namespace cpu;
TS_REGISTER_OPERATOR(Conv2DV2, GPU, name::layer::conv2d_v2())
TS_REGISTER_OPERATOR(Pooling2DV2, GPU, name::layer::pooling2d_v2())
TS_REGISTER_OPERATOR(DepthwiseConv2DV2, GPU, name::layer::depthwise_conv2d_v2())
