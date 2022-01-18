gatherv2(x..device, indices..device) ->y..device
描述：按对应索引提取tensor
输入：
x: Tensor 输入数据
indices: IntTensor  Int类型
索引的最后一个维度对应于元素 (如果 indices.shape[-1] == x.rank) 或切片 (如果 indices.shape[-1] < x.rank) 沿参数 indices.shape[-1] 中的维度.输出张量具有形状
indices.shape[:-1] + x.shape[indices.shape[-1]:]
输出：
y: Tensor



resize_nearest_neighbor(x..device, size..host) ->y..device
描述：使用最近邻插值调整images为size 

输入：
x: Tensor 输入数据
size:  Int[2],，新图像的大小 
输出：
y: Tensor

参数：
align_corners: Int default:0 如果为1，则输入和输出张量的4个角像素的中
心对齐,并保留角落像素处的值
dim: 指定要调整的开始维度，调整维度为dim,dim+1


rsqrt (x..device) -> y..device
描述：计算 x 元素的平方根的倒数
输入：
x: Tensor 输入数据
输出：
y: Tensor


maximum (x1..device, x2..device) -> y..device
描述：计算x 和 y 的最大值, 支持广播
输入：
x1: Tensor 输入数据
x2: Tensor 输入数据
输出：
y: Tensor


max (x..device) -> y..device
描述：计算x 和 y 的最大值
输入：
x: Tensor 输入数据
输出：
y: Tensor
参数：
dim: 支持负数
keep_dims: If is 1, retain reduced dimensions with length 1, default is 1
说明：
输出维度: keep_dims=1, [x.shape[:dim],1, x.shape[dim+1:]]
          keep_dims=0, [x.shape[:dim], x.shape[dim+1:]]



square(x..device) -> y..device
描述：计算 x 元素的平方
输入：
x: Tensor 输入数据
输出：
y: Tensor


range(start..host, limit..host, delta..host ) -> y..host
描述：创建一个数字序列,该数字开始于 start 并且将 delta 增量扩展到不包括 limit 的序列
输入：
start:  Scalar 生成序列开始数据
limit:  Scalar 生成序列结束数据, [start,limit)
delta:  Scalar 步长　
输出：
y: Array


exp(x..device) ->y..device
描述：y = e^x
输入：
x: Tensor 输入数据
输出：
y: Tensor


slice(x..device) -> y..device
描述：从原始输入数据中选择以begin开始的尺度大小为size的切片
输入：
x: Tensor 输入数据
输出：
y: Tensor
参数：
begin:IntArray
size: IntArray



argmax(x..device)-> y..device
描述：针对dim参数,去选取X中相对应轴元素值大的索引
输入：
x: Tensor 输入数据
输出：
y: IntTensor

参数：
dim: Int 指定dim:从x的哪个维度开始



expand_dims(x..device) -> y..device = delete
描述：在索引dim处添加1个大小的维度
输入：
x: Tensor 输入数据
输出：
y: Tensor

参数：
dim: IntScalar 指定dim:从x的哪个维度开始



non_max_suppression_v3(box..device, scores..device) ->y..device
描述：按照参数scores的降序贪婪的选择边界框的子集
输入：
box: Tensor[num_boxes, 4] 输入数据,2d坐标
scores: Float[num_boxes] 数组，
输出：
y: Tensor IntArray, 选出来的留下来的边框下标,下标必须大于等于0,数组大小为max_output_size, 如果下标为-1,仅为占位符

参数：
max_output_size: Int 最多可以利用NMS选中多少个边框, <= $num_boxes
iou_threshold:   Float IOU阙值展示的是否与选中的那个边框具有较大的重叠度
score_threshold:   Float A float representing the threshold for deciding when to remove boxes based on score.
mode: String Default["xyxy"] or ["xywh"]


topkv2(x..device [, number..host]) -> values..device, indices..device
描述：返回 x 中最后一个维度中的前nums个最大的数，并且返回它们所在位置的索引
输入：
x: Tensor 输入数据
输出：
y:    PackedTensor, 包含两个返回值:values和indices. 
values: Tensor,类型同x, 数据维度是 batch_size 乘上 k 个最大值。 
indices: IntTensor, 数据维度同values
参数：
number: Int 设置每行要返回的数据个数
sorted: Int 是否做排序,0：不要排序，1：需要排序,当前输出数据都是排序的


