# 算符支持

----

这里列举出，框架已经支持的算符，以及对应的参数设置。

所有内置算符都包含参数：
- `#op`: `String` 算符的名称
- `#name`: `String` 实例化名称
- `#output_count`: `Int` 输出结果数
- `#shape`: `IntArray` 大小  
- `#dtype`: `Int` dtype enumrate value

算符名称长度应该在 `[0, 32)`。

## 内置算符

### _field(a) -> field
描述：用于获取 `Packed Tensor` 的元素。  
输入：`a`: `Tensor` 压缩数据格式  
输出：`field` `Tensor`  
参数：

- `offset`: `Int` `[Required]` 要取输入 `Packed Tensor` 的偏移下标

### _pack(a) -> packed
描述：打包输入元素，输出 `Packed Tensor`  
输入：`a`: `List<Tensor>` 要进行打包的数据  
输出：`packed` `Tensor`  
参数：无

### _resize2d(x..device, size..host) -> y..device
描述：输入原图和要缩放的大小，输出缩放后的数据  
输入：`x`: `Tensor` 要缩放的数据  
输入：`size`: `IntArray` 数组的长度，要和 `x` 的维度一致。要缩放的图像的大小，包含`-1`表示扩展  
输出：`y`: `Tensor` 缩放后的图像   
参数：

- `type`: `Enum[linear=0, cubic=1, nearest=2, hard=3]` `[Optional] Default=linear` 图像缩放类型  

举例：  
输入: `Tensor[1, 640, 480, 3]` 和 `[-1, 300, 300, -1]`，输出 `Tensor[1, 300, 300, 3]`。

说明：  
`$x.shape.size == $size.size`  
`size`中，`-1`表示通道维度，例如`[-1, 300, 300]`，表示要将输入的`CHW`格式图像缩放到`(300, 300)`大小。
其中，不为`-1`的数值必须大于`0`，存在且只存在两个，两个值必须在相邻维度。
例如：`size` 为 `[-1, -1, 3]` 和 `[400, -1, 300]` 都是错误输入。

### _transpose(x..device) -> y..device
描述：对输入的 Tensor 进行维度转换，输出转换后的图像  
别名：`permute`  
输入：`x`: `Tensor` 要转换的数据  
输出：`y`: `Tensor` 转换后的数据  
参数：   
- `permute`: `IntArray` `[Optional]` 数组的长度，要和 `x` 的维度一致。输入的第 `i` 个维度就是 `permute[i]` 的维度。
如果，没有设置该参数，则相当于普通矩阵转置。

注意:  
如果`x.dims`比`permute.size`大的话，会扩展`x`的维度，使之与`permute.size`相同。

举例：  
如果 `x.shape` 为 `[1, 640, 480, 3]`，`permute` 为 `[0, 3, 1, 2]`，
输出 `b.shape` 为 `[1, 3, 640, 480]`。数据类型不变。
如果 `x.shape` 为 `[3]`，`permute` 为 `[0, 3, 1, 2]`，
输出 `b.shape` 为 `[1, 3, 1, 1]`。数据类型不变。

说明：  
`permute` 中的元素必须是有效维度下标，且每个下标有且必须出现一次。`tranpose` 会对存储格式产生影响。

### _reshape(x..device) -> y..device
描述：对输入的 Tensor 进行维度变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据  
输出：`y` `Tensor` 转换后的数据  
参数：   
- `shape`: `IntArray` `[Required]` 输出的 `shape` 要和此参数一致，中间可以出现最多一个 `-1` 表示维度填充，保证输出的元素个数和输入的元素个数一致。

举例：  
如果 `x.shape` 为 `[4, 2700]`，`shape` 为 `[-1, 300, 300, 3]`，
输出 `y.shape` 为 `[4, 300, 300, 3]`。数据类型不变。

说明：  
此操作只影响 `Tensor` 的 `shape` 不会对内存布局产生影响。


### dimshuffle(x..device) -> y..device
描述：对输入的 Tensor 进行 channel 变换，输出转换后的数据  
输入：`x` `Tensor` 要转换的数据    
输出：`y` `Tensor` 转换后的数据

参数：  
- `dim` `Int` 要进行 `shuffle` 的维度  
- `shuffle` `IntArray` 维度 `>= 1`，每个元素必须属于 `[0，rank(x, dim)]`

举例：  
如果 `x.shape` 为 `[300, 300, 3]` 那么 `_dimshuffle(x, 2, [2, 1, 0])`
完成了原始数据的第`2`个维度进行了按照 `shuffle` 进行了 shuffle。输出仍旧是 `[300, 300, 3]`。       

说明：  
`shuffle` 的维度可以比对应 `x` 的 `dim` 维度多或者少。从而完成通道复制的效果。
如果 `x` 的 `shape` 为 `[100, 100, 1]` 那么 `_dimshuffle(x, 2, [0, 0, 0])`
的输出 `shape` 为 `[100, 100, 3]`

### conv2d(x..device, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`w` `Tensor4D` `shape` 为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。默认`0` `padding`。  
输出大小的计算公式为：  
```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_w_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### transpose_conv2d(x..device, w..device) -> y..device
描述：对输入的 Tensor 进行 二维反卷积操作，输出反卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`w` `Tensor4D` `shape` 为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D`

注意：
以下所有参数为卷积参数，计算过程为逆过程

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。默认`0` `padding`。
输出大小的计算公式为：
```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_w_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### _shape(x..device) -> shape..host
描述：对输入的 `Tensor` 返回对应的 `shape`。  

说明：  
返回的 `shape` 默认在 `CPU` 存储上。

### pad(x..device, padding..host) -> y..device
描述：对输入的 `Tensor` 进行 `pad`。

参数：  
- `padding_value` `Scalar` `[Optional] Default=0` 表示 `padding` 时填充的参数

说明：  
其中 `padding` 的第一个维度和 `x` 的 `shape` 维度一致，第二个维度为常数 `2`。
输出的大小按照 `padding` 后的大小计算，最小为 `1`。


### depthwise_conv2d(x..device, w..device) -> y..device
描述：对输入的 Tensor 进行`Depthwise`二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`w` `Tensor4D` 格式为 `[multiplier_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D` 输出的 channel 数量为 `multiplier_channels * input_channels`。

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
输出大小在计算时，除法采用向下取整，最小为1。默认padding值为0。


### add_bias(x..device, b..device) -> y..device
描述：对输入的 Tensor 加上偏置。  
输入：`x` `Tensor` 输入数据  
输入：`b` `Tensor` 维度和通道数相同，通道的维度通过 `dim` 来指定。  
输出：`y` `Tensor`  

参数：  
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `dim` `Int` 通道所在的维度

说明：
`format` 和 `dim` 至少设置一项即可。


### conv2d_v2(x..device, padding..host, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据  
输入：`padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
   在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
   在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
输入：`w` `Tensor4D` 格式为 `[output_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
`type` 在当前版本中，固定为 `NCHW`。
输出大小在计算时，除法采用向下取整，最小为1。默认padding值为0。
输出大小的计算公式为：  
```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_w_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```


### depthwise_conv2d_v2(x..device, padding..host, w..device) -> y..device
描述：对输入的 Tensor 进行`Depthwise`二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据  
输入：`padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
   在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
   在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
输入：`w` `Tensor4D` 格式为 `[multiplier_channels, input_channels, kernel_height, kernel_width]`
输出：`y` `Tensor4D` 输出的 channel 数量为 `multiplier_channels * input_channels`。

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding_value` `Scalar Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
输出大小计算除法时，向下取整，最小为1。默认0padding。

### batch_norm(x..device, mean..device, variance..device) -> y..device
描述：单纯进行 BN
输入：`x`: `Tensor4D`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`   
输出：`y`: `Tensor4D` `$y.shape == $x.shape` 

参数：
- `dim`: `Int` 表示`channel`所在的维度
- `epsilon`: `Scalar Default(1e-5)` 表示约束系数，作用见说明

说明：
`y = (x - mean) / (sqrt(variance + epsilon))`

### batch_scale(x..device, scale..device, bias..device) -> y..device
描述：单纯进行 Scale
输入：`x`: `Tensor4D`  
输入：`scale`: `Array` `$mean.size == $x.shape.size`  
输入：`bias`: `Array` `$mean.size == $x.shape.size`    
输出：`y`: `Tensor4D` `$y.shape == $x.shape`

参数：
- `dim`: `Int` 表示`channel`所在的维度

说明：
`y = x * scale + bias`

### fused_batch_norm(x..device, mean..device, variance..device, scale..device, bias..device) -> y..device
描述：等价于 `batch_scale(batch_norm(mean, variance), scale, bias)`  
输入：`x`: `Tensor4D`  
输入：`mean`: `Array` `$mean.size == $x.shape.size`  
输入：`variance`: `Array` `$mean.size == $x.shape.size`  
输入：`scale`: `Array` `$mean.size == $x.shape.size`  
输入：`bias`: `Array` `$mean.size == $x.shape.size`  
输出：`y`: `Tensor4D` `$y.shape == $x.shape`

参数：  
包含 `batch_scale`、`batch_norm` 参数。

### add(x..device, a..device) -> y..device
描述：进行矩阵加法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i + a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### sub(x..device, a..device) -> y..device
描述：进行矩阵减法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i - a_i`，要求`a`和`x`的维度一样，或者为`1`。  
关于广播的含义见附1。

### mul(x..device, a..device) -> y..device
描述：进行矩阵乘法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i * a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### div(x..device, a..device) -> y..device
描述：进行矩阵除法，支持 `Broadcast`  
输入: `x`: `Tensor`  
输入: `a`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
无

说明：
`y_i = x_i / a_i`，要求`a`和`x`的维度一样，或者为`1`。 
关于广播的含义见附1。

### inner_prod(x..device, a..device) -> y..device
描述：进行内积操作，即`y = x \mul a`  
输入: `x`: `Matrix`  
输入: `a`: `Matrix`  
输出: `y`: `Matrix`  

参数: `transpose` `bool Default=false`, 控制`a`的转置.

注意: 如果`x.dims > 2`, 则`flatten(x) \dot a`.

### relu(x..device) -> y..device
描述：激活函数`relu`,  `y = x > 0 ? x : 0`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

### relu_max(x..device) -> y..device
描述：激活函数`relu_max`,  `y = min(x > 0 ? x : 0, max)`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：
- `max`: `Scalar` 输出的最大值。

### sigmoid(x..device) -> y..device
描述：激活函数`sigmoid`,  `y = 1 / (1 + exp(-x))`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

### prelu(x..device, slope..device) -> y..device
描述：激活函数`prelu`, `y = x > 0 ? x : slope * x`  
输入: `x`: `Tensor`  
输入: `slope`: `Array` 维度与`dim`给定的维度相同  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：
- `dim`: `Int` slope 所在的维度，此参数必须设置。


### leaky_relu(x..device) -> y..device
描述: 激活函数`leaky_relu`, `y = x > 0 ? x : scale * x`  
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数:  
- `scale`: `Float`


### softmax(x..device) -> y..device
描述：对Tensor进行softmax操作。
输入: `x`: `Tensor`  
输出: `y`: `Tensor` `$y.shape == $x.shape`  

参数：  
- `dim`: `Int` softmax 要处理的维度，此参数必须设置。 
- `smooth`: `bool Default=false` 

说明：  
smooth 为`false`时：
```
y_i = exp(x_i) / \sum{exp(x_i)}
```
smooth 为`true`时：
```
t_i = x_i - max(x)
y_i = exp(t_i) / \sum{exp(t_i)}
```


### concat(x..device) -> y..device
描述：链接算符，用于对数据的拼接。 
输入: `x`: `List<Tensor>` 要进行拼接的数据  
输出: `y`: `Tensor`  

参数：
- `dim`: `Int` 要进行Concat的维度。

说明：  
要把输入的元素进行拼接，在`dim`维度上，除了`dim`维度，其他输入数据的维度必须相同。
输出的`dim`维度是输入对应`dim`维度的和。


### stack(x..device) -> y..device
描述:  堆叠算符，用于对数据的堆叠。
输入: `x`: `List<Tensor>`
输出: `y`: `Tensor`

参数:  
- `axis`: `Int` Stack时的轴线.

注意:  
等同于`numpy.stack`

### pooling2d(x..device) -> y..device
描述：进行下采样

输入: `x`: `Tensor`
输出: `y`: `Tensor`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_type` `Enum[black=0, copy=1, loop=2] Default(0)` `[Optional]` 表示 `padding` 时填充的参数
- `ksize` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

说明：  
计算大小时，除法的结果采用向上取整，最小为1。
pooling size计算公式:

```c
output_h = ceil((input_h + pad_h_up + pad_h_down - kernel_h) / static_cast<float>(stride_h) + 1);
output_w = ceil((input_w + pad_w_left + pad_w_right - kernel_w) / static_cast<float>(stride_w) + 1);
```
`padding_type`为`black`时，超出可计算区域的结果为0。

### pooling2d_v2(x..device, padding, ksize, stride)
描述：进行下采样

输入：
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `ksize` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

参数：

- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1] Default max` `[Optional]`
- `padding_type` `Enum[black=0, copy=1, loop=2] Default black` `[Optional]` 表示 `padding` 时填充的参数

说明：  
计算大小时，除法的结果采用向上取整，最小为1。
`padding_type`为`black`时，超出可计算区域的结果为0。

### flatten(x) -> y
描述：把输入的shape，调整成2维的矩阵。  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
- `dim`: `Int` `[Optional] Default=1`

说明：  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，输出的 `shape` 为 `[1, 180]`。  
在对应 `dim` 位置进行拉伸。  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，`dim = 2`，输出的 `shape` 为 `[1, 20, 9]`。  
输入 `x` 的 `shape` 为 `[2, 3]`，`dim = 2`，输出的 `shape` 为 `[2, 3, 1]`。  

### flatten2d(x) -> y
描述：把输入的shape，调整成2维的矩阵。
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

- `dim`: `Int` `[Optional] Default=1`

说明：  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，输出的 `shape` 为 `[1, 180]`。  
在对应 `dim` 位置进行拉伸。  
输入 `x` 的 `shape` 为 `[1, 20, 3, 3]`，`dim = 2`，输出的 `shape` 为 `[20, 9]`。  
输入 `x` 的 `shape` 为 `[2, 3]`，`dim = 2`，输出的 `shape` 为 `[6, 1]`。  
输入 `x` 的 `shape` 为 `[2, 3]`，`dim = 0`，输出的 `shape` 为 `[1, 6]`。  

## to_float(x) -> y
描述：把输入类型，调整成 `float` 类型。  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

## prewhiten(x) -> y
描述：进行图像白化  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

说明：  
对于输入的每一个样本 `x_i` 执行：
```cpp
template <typename T>
void prewhiten(T *data, size_t len)
{
	double mean = 0;
	double std_dev = 0;
	T *at= nullptr;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) mean += *at;
	mean /= len;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) std_dev += (*at - mean) * (*at - mean);
	std_dev = std::sqrt(std_dev / len);
	std_dev = std::max<T>(std_dev, 1 / std::sqrt(len));
	double std_dev_rec = 1 / std_dev;

	at = data;
	for (size_t i = 0; i < len; ++i, ++at) {
		*at -= mean;
		*at *= std_dev_rec;
	}
}
```

### _cast(x..device) -> y
输入：`x`: `Tensor` 
输出：`y`: `Tensor`

参数：  
- `dtype` `Int` 要转换成的类型

### gather(x..device, indices..host) -> y
输入：`x`: `Tensor`
输入：`indices`:  ` IntArray`
输出：`y`: `Tensor`

参数：  
- `axis` `Int` `[Optional] Default=0` 要进行gather的维度。

说明：  
等价于`numpy.take(x, indices, axis=axis)`

### unsqueeze(x..device) -> y
输入：`x`: `Tensor`
输出：`y`: `Tensor`

参数：  
- `axes` `IntArray` 要填充维度

说明：  
等价于`numpy.expend_dims(x, axis) for axis in axes`


### _reshape_v2(x..device, shape..host) -> y..device
描述：对输入的 Tensor 进行维度变换，输出转换后的数据  
输入：`x`: `Tensor` 要转换的数据  
输入：`shape`: `IntArray` 输出的 `shape` 要和此参数一致，中间可以出现最多一个 `-1` 表示维度填充，保证输出的元素个数和输入的元素个数一致。  
输出：`y`: 转换后的数据  

举例：  
如果 `x.shape` 为 `[4, 2700]`，`shape` 为 `[-1, 300, 300, 3]`，
输出 `y.shape` 为 `[4, 300, 300, 3]`。数据类型不变。

说明：  
此操作只影响 `Tensor` 的 `shape` 不会对内存布局产生影响。


### gemm(a..device, b..device, c..device) -> Y..device
描述：就是GEMM，嗯。  
输入：`A` `Matrix`
输入：`B` `Matrix`
输入：`C` `Matrix` 或者可以广播的 `Tensor` 
输入：`Y` `Matrix`

参数：
- `alpha` `Float` `[Optional] Default=1.0f`
- `beta` `Float`  `[Optional] Default=1.0f`
- `transA` `Int`  `[Optional] Default=false` 布尔变量
- `transB` `Int`  `[Optional] Default=false` 布尔变量

说明：
```
A' = transpose(A) if transA else A
B' = transpose(B) if transB else B
Compute Y = alpha * A' * B' + beta * C,
where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K),
input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N).
A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
This operator supports unidirectional broadcasting
(tensor C should be unidirectional broadcastable
to tensor A * B); 
```

### lrn (x..device) -> y..device = delete

参数：
- `dim` `Int` 要进行LRN的维度
- `alpha` `Float` `Default(0.0001)`
- `beta` `Float` `Default(0.75)`
- `bias` `Float` `Default(1)`
- `size` `Int` `Required`

说明：  
按照 `LRN` 的传统公式和做法


### batch_to_space4d(x..device) -> y..device

输入: `x`: `Tensor`
输出: `y`: `Tensor`

参数：

- `crop` `Int[2, 2]` `[[crop_top, crop_bottom], [crop_left, crop_right]] `
- `block_shape` `Int[2]` `[block_height, block_width]`

说明：
这里的操作为NCHW算符。
见：[BathToSpace](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-bnyg2ckl.html)


### space_to_batch4d(x..device) -> y..device

参数：
- `padding` `Int[2, 2]` `[[padding_top, padding_bottom], [padding_left, padding_right]] `
- `block_shape` `Int[2]` `[block_height, block_width]`

说明：
这里的操作为NCHW算符。
见：[BathToSpace](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-emqk2kf4.html)


### global_pooling2d(x)
描述：进行全局下采样

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `type` `Enum[max=0, avg=1]` 

说明：  
输出大小固定为1x1。

### _limit(x..device) -> y..device
描述：对blob大小进行限制，如果输入大于这个大小，则进行center_crop，
否则保持原有大小。

参数：  
- `shape`: `IntArray` 输出限制，维度要小于x的维度。-1表示不进行限制。

说明：  
假如shape小于x的大小，则在shape高位补充-1，直到维度相同，再进行调整。

### shape_index_patch(x..device, pos..device) -> y..device
描述：根据pos在x上进行采样。  
输入：`x`: `Tensor4D` shape 为 `[number, channels, height, width]`  
输入：`pos`: `Tensor4D` shape 为 `[number, landmark, 1, 1]`  

输出：`y`: `Tensor5D` shape 为 `[number, channels, x_patch_h, landmark / 2, x_patch_w]`
其中 `x_patch_h = int(origin_patch.h * x.height / origin.h + 0.5)`,  
`x_patch_w = int(origin_patch.w * x.width / origin.w + 0.5)`,  
注意: 这是对应某一个实现的版本。


参数：  
- `origin_patch`: `Int[2]{h, w}`  
- `origin`: `Int[2]{h, w}`  

说明：  
`pos.number == x.number`，根据pos表示的位置信息，在对应位置crop出`[x_patch_h, x_patch_w]`大小。

### affine_sample2d(x..device, size..host, affine..host) -> y..device

描述：根据affine，在x上采样出大小为size的图像
输入：`x`: `Tensor`
输入：`size`: `Int[2]` 表示2d的采样大小, {height, width}
输入：`affine`: `Float[3, 3]` 仿射变换矩阵

参数：
- `type`: `Enum[linear=0, cubic=1, nearest=2, hard=3] Default linear`
- `dim`: `Int Default -2`
- `outer_value`: `Optional Float Default [0]`

说明：
y的坐标为`[x, y]`映射到原图为`affine * [x, y, 1]'`，然后根据type进行采样。
这里坐标全部为列向量。
`dim` 和 `dim+1` 表示了图像的二维采样。

### sample2d(x..device) -> y..device

描述：根据affine，在x上采样出大小为size的图像
输入：`x`: `Tensor`

参数：
- `type`: `Enum[linear=0, cubic=1, nearest=2]` `[Optional] Default=nearest`
- `dim`: `Int` `[Optional] Default=-2`
- `scale`: `Float`

说明：
`scale > 1` 表示上采样，`scale < 0` 下采样。
`dim` 和 `dim+1` 表示了图像的二维采样。


### sample2d_v2(x..device, scale..host) -> y..device
描述: 内部调用`resize2d(x, x.shape * scale)`.  
输入: `x`: `Tensor`  
输入: `scale`: `FloatArray`  

参数:  
- `type`: `Enum[linear=0, cubic=1, nearest=2, hard=3] Default hard`

说明:  
`x.shape.dim == scale.dim`


### sample2d_like(x..device, y..device) -> y..device = delete
描述: 内部调用 `resize2d(x, y.shape)`.  
输入: `x`: `Tensor`  
输入: `y`: `Tensor`  

参数:  
- `type`: `Enum[linear=0, cubic=1, nearest=2, hard=3] Default hard`
说明:  
Resize x to y's shape, y must have no more the 2 different dim to x. 
将x调整为y的形状，y不得超过x的2个不同维度


### chunk(x..device) -> y..device

描述: concat的逆操作
输入: `Tensor`
输出: `Packed Tensor`

参数：
- `chunks`: `Int`  要拆分的个数
- `dim`: `Int` `[Optional] Default=-2` 要拆分的坐标

### dcn_v2_forward(x..device, w..device, b..device, offset..device, mask..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x` `Tensor4D` 输入数据
输入：`offset` `Tensor4D`
输入：`mask` `Tensor4D`
输入：`w` `Tensor4D` 和卷积同样参数
输入：`b` `Tensor4D` 和卷积同样参数
输出：`y` `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`，这里只支持NCHW
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation` `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `deformable_group`

说明：
参考代码：
[DCNv2](https://github.com/CharlesShang/DCNv2)

`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。默认`0` `padding`。
输出大小的计算公式为：
```
Waitting for sure
```

### mean(x..device) -> y..device = delete
参数：
- `reduction_indices` `IntArray` 选择要进行平均数计算的维度。
- `keep_dims` `Int` `Default=1`， 布尔值，表示是否保留维度
说明：  
返回`reduction_indices`维度内的平均数  
绝大多数情况优先采用global_average_pooling.

### reduce_sum(x..device) -> y..device
参数：  
- `dims` `Int` 要进行求和的维度  
- `keep_dims` `Boolean` `[Optional] Default=1` 是否保留求和后的位置  

说明：  
返回在对应维度上分别求和的结果。

### reduce_mean(x..device) -> y..device
参数：  
- `dims` `Int` 或 `IntArray` 要进行求和的维度，如果为 `IntArray` 则要求维度是连续的  
- `keep_dims` `Boolean` `[Optional] Default=1` 是否保留求和后的位置  

说明：  
返回在对应维度上分别求均值的结果。


### squeeze(x..device) -> y

参数：  
- `axes` `IntArray` `OPTIONAL` 要去除的维度

说明：  
如果axes为空，则删除所有为1的维度。


### _nhwc_resize2d(x..device) = delete

参数：  
- `size` `Int[2]` 内容为 `{width, height}`。

### _nhwc_crop2d(x..device) = delete

参数：  
- `rect` `Int[4]` 内容为 `{x, y, width, height}`。

### _nhwc_center_crop2d(x..device)

参数：  
- `size` `Int[2]` 内容为 `{width, height}`。

### _nhwc_scale_resize2d(x..device)
参数：
- `size` `Int[1]` 或 `Int[2]` 内容为 `{width, height}`
- `type`: `Enum[linear=0, cubic=1]`  `[Optional] Default=linear` 图像缩放类型  

说明：  
如果输入为单个int值，则将输入图像的短边resize到这个int数，长边根据对应比例调整，图像长宽比保持不变。  
如果输入为(w,h)，且w、h为int，则直接将输入图像resize到(w,h)尺寸，图像的长宽比可能会发生变化


### _nhwc_letterbox(x..device)
参数：
- `size`: `Int[1]` 或 `Int[2]`
- `type`: `Enum[linear=0, cubic=1]` `[Optional] Default=linear`图像缩放类型  
- `outer_value`: `Float`  `[Optional] Default=0` 采样图像外区域的值  

说明：  
如果`outer_value`不设置，则采样不到取最近邻的值。否则用`outer_value`填充每个单元。
`size`为`{width, height}`格式。


### divided(x..device) -> y..device
参数：  
- `size`: `IntArray` 表示了每个维度向上去整的值。
- `padding_value`: `Float` `[Optional]` 填充值。

说明：  
把x的大小调整为可以被size序列整除的大小。
size大小和x.dims相同，表示了每一个维度的大小。


## yolo(x..device) -> y..device, classes, mask, anchors
描述：等价于darknet.yolo 层(CPU)的计算  
输入：`x`: `Tensor4D`
输出：`y`: `PackedTensor`

参数：  
- `classes`: `Int` 分类数  
- `mask`: `IntArray`   
- `anchors`: `FloatArray`  


## yolo_poster(x, yolo...) -> y.host
描述：进行yolo的后处理，输入为每个yolo层的输出。输出检测的框。
输入：`x`: 输入节点，用于获取网络输入的形状。
输入：`yolo`: 可变长度的yolo层的输入。

参数：  
- `thresh`: `Float` 置信度阈值。  
- `nms`: `Float` NMS 的阈值。

说明：  
输出格式为：N * M, 其中N为输出结果数量，
其中M 为 4(box: x, y, w, h) + 1(prob) + 1(label)
矩形表示为`[0, 1]`区间的值。


### _nhwc_channel_swap(x..device) = delete

参数：  
- `shuffle` `IntArray`

### _nhwc2nchw(x..device) = delete

参数：无

### _tf_conv2d_padding(x..device, w..device) -> dynamic_padding

参数：
- `format`: `String` 为 `NCHW` 或者 `NHWC`
- `padding_method`: `String` 表示 `padding` 方式为`SAME` 或 `VALID`
- `stride`: `Int[4]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dilation`: `Int[4]` `[Optional]` `batch` 和 `channels` 的默认为 `1`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding`: `Int[4, 2]` `[Optional]` `Default=Zero padding` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

### _tf_pooling2d_padding（x, ksize, stride) -> dynamic_padding
描述：  
- `x`: `Tensor4D` 预计要进行 padding 的数据
- `ksize`: `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride`: `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`:  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `format`: `String` 为 `NCHW` 或者 `NHWC`
- `padding_method`: `String` 表示 `padding` 方式为`SAME` 或 `VALID`
- `padding`: `Int[4, 2]` `[Optional]` `Default=Zero padding` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。


### mx_conv2d_padding = delete

### _mx_pooling2d_padding(x, ksize, stride) -> dynamic_padding
描述：  
- `x`: `Tensor4D` 预计要进行 padding 的数据
- `ksize`: `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride`: `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`:  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `format`: `String` 为 `NCHW` 或者 `NHWC`
- `valid`: `Int`

非0数表示计算为：
```
output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
```
0表示计算为：
```
output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
```
- `padding` `Int[4, 2]` 静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

### _onnx_pooling2d_padding(x, ksize, stride) -> dynamic_padding
描述：  
- `x` `Tensor4D` 预计要进行 padding 的数据
- `ksize` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `stride` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `dynamic_padding`  `Int[4, 2]`输出的padding形式，为4x2维

参数：  
- `auto_pad` `String` 为 `NOTSET`、`SAME_UPPER`、`SAME_LOWER`、`VALID`. `[Optional] Default=NOTSET`
    `NOTSET`表示计算为：
    ```
    output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
    其中pad_shape[i]是沿i为轴的pads之和
    ```
    `VALID`表示计算为：

    ```
    output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
    ```
    `SAME_UPPER`和`SAME_LOWER`表示计算为：
    ```
    output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    ```

动态padding大小为：
```
pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
```
- `padding` `Int[4, 2]` `[Optional] Default=0`静态进行padding的数据
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。

### winograd_transform_kernel(x) -> y
描述：将卷积的kernel参数变换为winograd f63或者f23的kernel参数并pack_A(pack矩阵乘法左值)。  
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

参数：  
- `winograd_mode` `String` 为 `winograd_f63` 或者 `winograd_f23` `[Optional] Default=winograd_f63`

说明：  
输入 `x` 的 `shape` 为 `[output_channels, input_channels, 3, 3]`
`winograd_mode`为`winograd_f63`时,输出的 `shape` 为 `[output_channels, input_channels, 8, 8]`。
`winograd_mode`为`winograd_f23`时,输出的 `shape` 为 `[output_channels, input_channels, 4, 4]`。
每一个通道上的计算公式：U = GgGT,其中g为kernel的一个通道数据,G为变换矩阵

```
        if(winograd_mode == winograd_f63){
            const T G[8][3] = {
                { T(1),     0,     0 },
                { -T(2) / 9,  -T(2) / 9,  -T(2) / 9 },
                { -T(2) / 9,   T(2) / 9,  -T(2) / 9 },
                { T(1) / 90,  T(1) / 45,  T(2) / 45 },
                { T(1) / 90, -T(1) / 45,  T(2) / 45 },
                { T(1) / 45,  T(1) / 90, T(1) / 180 },
                { T(1) / 45, -T(1) / 90, T(1) / 180 },
                { 0,     0,     1 }
            };
        }
        else if(winograd_mode == winograd_f23){
            const T G[12] = {
                1,     0,     0,
                T(1) / 2,   T(1) / 2,   T(1) / 2,
                T(1) / 2,   -T(1) / 2,   T(1) / 2,
                0,     0,     1
            };
        }
        
        U = GgGT
        y = pack_A(temp)
```
### conv2d_winograd(x..device, padding..host, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x`: `Tensor4D` 输入数据
输入：`padding`: `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
   在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
   在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
输入：`w`: `Tensor4D` `shape` 为 `[output_channels, input_channels, kernel_height, kernel_width]` or `[output_channels, input_channels, 4, 4]`,`[output_channels, input_channels, 8, 8]`,
输出：`y`: `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar` `[Optional] Default=0`  表示 `padding` 时填充的参数
- `kernel_winograd_transformed` `Bool` `[Optional] Default=false` 表示kernel未被transformed

说明：
包含两种`winograd_mode`:`winograd_f63`,`winograd_f23`根据通道数以及输入size决定使用哪种模式.
`kernel_winograd_transformed`为`true`时,输入`w` `shape` 为`[output_channels, input_channels, 4, 4](winograd_f23)` or `[output_channels, input_channels, 8, 8](winograd_f63)`
conv2d_winograd要求dilation,stride均为1.
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。  
输出大小的计算公式与卷积相同,注意这里的`kernel size`为未变换前的`kernel size`即`kerne_h=3`,`kernel_w=3`,`pad`均为`0`：  

```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_w_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### conv2d_winograd_v2(x..device, w..device) -> y..device
描述：对输入的 Tensor 进行 二维卷积操作，输出卷积后的数据
输入：`x`: `Tensor4D` 输入数据
输入：`w`: `Tensor4D` `shape` 为 `[output_channels, input_channels, kernel_height, kernel_width]`or `[output_channels, input_channels, 4, 4]`,`[output_channels, input_channels, 8, 8]`
输出：`y`: `Tensor4D`

参数：
- `format` `String` 为 `NCHW` 或者 `NHWC`
- `padding` `Int[4, 2]` `[Optional]` `batch` 和 `channels` 的默认为 `[0, 0]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。
- `padding_value` `Scalar` `[Optional] Default=0`  表示 `padding` 时填充的参数
- `kernel_winograd_transformed` `Bool` `[Optional] Default=false` 表示kernel未被transformed

说明：
包含两种`winograd_mode`:`winograd_f63`,`winograd_f23`根据通道数以及输入size决定使用哪种模式.
`kernel_winograd_transformed`为`true`时,输入`w` `shape` 为`[output_channels, input_channels, 4, 4](winograd_f23)` or `[output_channels, input_channels, 8, 8](winograd_f63)`
conv2d_winograd要求dilation,stride均为1.
`type` 在当前版本中，固定为 `NCHW`。
输出大小计算除法时，向下取整，最小为`1`。  
输出大小的计算公式与卷积相同,注意这里的`kernel size`为未变换前的`kernel size`即`kerne_h=3`,`kernel_w=3`,`pad`均为`0`：  

```
pad_h = pad_h_top + pad_h_bottom
pad_w = pad_w_left + pad_w_right
output_h = floor((height + pad_h -
			(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
output_w = floor((width + pad_w -
			(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
```

### l2_norm(x..device) -> y..device
描述：在dim维度上执行L2范数。
参数：
- `dim` `Int` `[Optional] Default=-1`
- `epsilon` `Float` `[Optional] Default=1.00000001e-10f`

说明：  
For 1-D NDArray, it computes:  

```
out = data / sqrt(sum(data ** 2) + eps)
```
For N-D NDArray, if the input array has shape (N, N, ..., N),
```
for dim in 2...N
  for i in 0...N
    out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
```

### _dims(x..device) -> y..host
描述：返回输入的维度信息，即`x.dims()`


### _dtype(x..device) -> y..host = delete
描述：Return x.dtype()


### _cast_v2(x..device, dtype..host) -> y = delete
输入: `x` `Tensor`  
输入: `dtype` `Int` 
输出: `y` `Tensor` `dtype` 由参数`dtype`确定 


### _expand(x..device, dims..host) -> y..device
描述：如果`dims <= x.dims()`，返回`x`; 否则当`inverse`为`False`时，扩充`x`的形状为首先在形状前面插入后的维度。

参数:
- `front` `Int` `[Optional] Default=-1` 可以被添加在前面的最大维度
- `end` `Int` `[Optional] Default=-1` 可以被添加在后面的最大维度
- `inverse` `Bool` `[Optional] Default=false` 当`inverse` 为 `False`时，首先在前面添加维度；反之相反。

### tanh(x..device) -> y..device
描述：返回 `\frac{exp(x)-exp(-x)}{exp(x)+exp(-x)}`, 与`2 * sigmoid(2 * x) - 1`相等.

### abs(x..device) -> y..device
描述：返回输入的绝对值，即 `abs(x)`


### force_gray(x..device) -> y..device
描述：将图像转换为灰度模式并返回。假设通道是最后一维。

输入: `x`: `Tensor` 输入图像.
输出: `y`: `Tensor`

参数:  
- `scale` `FloatArray` `[Optional]` 对于`BGR`格式`Default=[0.114, 0.587, 0.299]`

说明:  
`x.shape[-1]` 等于`map`的`dims`或者1


### force_color(x..device) -> y..device
描述：将图像转换为彩色模式并返回。假设通道是最后一维。

输入: `x`: `Tensor` 输入图像.
输出: `y`: `Tensor`


### convert_color(x..device, code..host) -> y..device = delete
输入: `x`: `Tensor`输入图像.
输入:  `code`:  `Int`

输出: `y`: `Tensor`

### norm_image(x) -> y
描述：使用均值和标准差来归一化图像。
输入: `x`: `Tensor`  
输出: `y`: `Tensor`  

参数:
- `epsilon` `Float` `[Optional] Default=1e-5f`

说明：  
对于每个 `x_i \in x[i,:,:,:]` 进行如下操作：

```
x_i = (x_i - mean(x_i)) / (std_dev(x_i) + epsilon)
```

### sqrt (x..device) -> y..device
描述：计算 x 元素的平方根  
输入：`x`: `Tensor` 输入数据  
输出：`y`: `Tensor`  


### tile (x..device) -> y..device
描述：与 numpy.tile 等价  
输入: `x`: `Tensor`  
输出：`y`: `Tensor`  

参数：  
- `repeats` `IntArray` 长度和`x.shape`相同。

说明：  
`y = numpy.tile(x, repeats)`


### broadcast (x..device, shape..host) -> y..device
描述: 与`x * numpy.ones(shape)`相同


### proposal (list[scores]..device, prob, bbox, im_info) -> proposals..device
描述: 见 [Dragon proposal](http://dragon.seetatech.com/api/python/contents/operators/contrib/rcnn.html)  
输入: `list[scores]`: `List[Tensor]` 张量列表, 长度与`strides`一致  
输入: `prob`: `DTypeTensor`  
输入:  `bbox`: `DTypeTensor`  
输入:  `im_info`: `Float[3]`  
输出: `y`: `FloatTensor`

参数:  
~~`inputs` (sequence of Tensor) 输入.~~
- `strides` `IntArray`   `anchors` 的 `strides`.
- `ratios` `FloatArray`   `anchors` 的 `ratios`.
- `scales` `FloatArray`  `anchors` 的 `scales`.
- `pre_nms_top_n` `Int` `[Optional] Default=6000` `nms` 操作前 `anchor` 的数量
- `post_nms_top_n` `Int` `[Optional] Default=300` `nms` 操作后 `anchor` 的数量
- `nms_thresh` `Float` `[Optional] Default=0.7` `nms` 的阈值.
- `min_size` `Int` `[Optional] Default=16` `anchors` 的最小尺寸.
- `min_level` `Int` `[Optional] Default=2`   FPN 金字塔的顶层.
- `max_level` `Int` `[Optional] Default=5`   FPN 金字塔的底层.
- `canonical_scale` `Int` `[Optional] Default=224`映射策略的基准缩放尺度 The baseline scale of mapping policy.
- `canonical_level` `Int` `[Optional] Default=4` 规范缩放尺度的经典级别 Heuristic level of the canonical scale.



### roi_align (features..device, proposal..device) -> region..device
描述: 见 [Dragon ROIAlign](http://dragon.seetatech.com/api/python/contents/operators/vision.html#dragon.operators.vision.ROIAlign)  
输入: `features`: `DTypeTensor`  
输入: `proposal`: `FloatTensor`  
> features and proposal are same shape

输出: `region`: `Tensor`  

参数:  
~~ - `inputs` (sequence of Tensor) – The inputs, represent the Feature and RoIs respectively.~~
- `pool_h` `Int` `[Optional] Default=0`  被池化`tensor`的高.
- `pool_w` `Int` `[Optional] Default=0`  被池化`tensor`的宽.
- `spatial_scale` `Float` `[Optional] Default=1.0` 输入`tensor`上总下采样倍数的倒数.
- `sampling_ratio` `Int` `[Optional] Default=2` 每个RoI分组的采样网格数.


### _dragon_pooling2d_padding(x, ksize, stride) -> dynamic_padding
描述：  
输入: `x` `Tensor4D` 预计要进行 padding 的数据  
输入: `ksize` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。  
输入: `stride` `Int[4]`
在 `NCHW` 四个维度分别表示 `[batch, channels, height, width]`,
在 `NHWC` 四个维度分别表示 `[batch, height, width, channels]`。  
- `dynamic_padding`  `Int[4, 2]`输出的padding形式，为4x2维  
- `ceil` `Bool`计算输出大小时，是否使用ceil  

参数：  
- `auto_pad` `String` 为 `SAME_UPPER`、`SAME_LOWER`、`VALID`  
更多细节，见 `Dragon`.

### tile_v2 (x..device, repeats..host) -> y..device = delete


### slice_v2(x..device, begin..host, size..host) -> y..device
描述:  得到 `x[begin:begin+size)` 的 `slice`
输入:` x`: `Tensor`  
输入: `begin`: `IntArray`  
输入: `size`:` IntArray`  
输出: `y`: `Tensor`  

注意:  
内部调用`slice`  
~~if len(begin) or len(size) < x.dims, expand with (0, -1) as (begin, size).
`-1` means max size can be sliced.~~


### slice_v3(x..device, starts..host, ends..host[, axes..host[, steps..host]]) -> y..device
描述: `onnx.Slice`

### ceil(x..device) -> y..device
描述: `y = ceil(x)`

### _transpose_v2(x..device, permute..host) -> y.device
描述：对输入的 Tensor 进行维度转换，输出转换后的图像  
别名：`permute`  
输入：`x`: `Tensor` 要转换的数据  
输出：`y`: `Tensor` 转换后的数据  
参数：   
- `permute`: `IntArray` `[Optional]` 数组的长度，要和 `a` 的维度一致。输入的第 `i` 个维度就是 `permute[i]` 的维度。如果没有设置该参数，则相当于普通矩阵转置。

## 附录

1. 在做基础运算的时候，`x`和`a`有会三种意义，分别为`标量`，`张量`和`广播张量`。这里的广播张量的意义为：
   加入 x 的 shape 为 `[10, 10]` `a` 的 `shape` 为 `[10, 1]`，这个时候
   `y_{ij} = x_{ij} + a_{i0}` 在 `a` 的第二个维度，实现了广播。  
   注意： 广播的维度可以在矩阵中存在多份，默认维度大小为 `1` 的都支持广播。
   
2. `Pooling` 框架支持的 `padding` 类型。  
   tf_valid:
   ```
   output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h + 1) / (float)m_stride_h);
   output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w + 1) / (float)m_stride_w);
   ```
   tf_same:
   ```
   output_height = ceil((input_height + 2 * m_pad_h) / (float)m_stride_h);
   output_width = ceil((input_width + 2 * m_pad_w) / (float)m_stride_w);
   ```
   caffe(mx_same):
   ```
   output_height = ceil((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = ceil((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```
   mx_valid(tf_valid):
   ```
   output_height = floor((input_height + 2 * m_pad_h - m_kernel_h) / (float)m_stride_h + 1);
   output_width = floor((input_width + 2 * m_pad_w - m_kernel_w) / (float)m_stride_w + 1);
   ```

