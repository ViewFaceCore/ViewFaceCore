# 模块文件描述
> 该文件用于说明保存的模块文件的描述格式
> update in 2020-02-03, add string limit.

## 1. 声明定义

模型文件中数据都是以**小端**编码存储的。  

`字段`：首先定义了描述格式，以尖括号扩起来表示一个二进制的字段，形如`<$type[:$name]>`。
其中`$type`表示该字段的类型，`$size`表示该字段的名称。
例如：  
`<int32:length>`  
表示`length`字段是占用四个字节的整数，即`int32`。

`字段数组`：对于数组字段采用`<$type*$size[$name]>`表示，其中`size`表示重复字段的个数。
例如：
`<int32*4:size>`
表示`size`字段是含有4个连续的`int32`。

`字段反射`，根据字段名称可以获取到字段实际表示的内容，`$size`表示`size`字段存储的内容。
特殊地，当字段为数组时`$array[0]`表示数组`$array`的第`0`个元素；
当字段为结构体时`$struct.field`表示`$struct`结构体的`field`成员。

`结构体`：对于连续存储的结构体字段，不使用连接符直接连续书写，例如：  
`<int32:shape_size><int32*$shape_size:shape>`  
表示了首先存储了四字节的整数表示字段`shape_size`，然后存储了`$shape_size`个4四字节整数。

`结构体数组`：对于复杂类型的定义，在不引起歧义的情况下，可以直接书写。或者采用方括号括起来。
一个完整的复杂字段，可以当做单独的字段使用。
例如：
`<<string><tensor>*size:params>`或`<[<string><tensor>]*size:params>`  
表示存储了连续`size`个`<string><tensor>`。

`类型声明`：对于复杂的类型可以`类型声明`，采用`$lhs:=$rhs;`符号表示，例如：  
`string := <int32:length><char8*$length>;`  
定义了一个新类型字符串的二进制格式。

`表达式`：数学表达式采用圆括号括起来，表示数学计算。例如：  
`(prob(size))`  
表示对数组`size`进行求乘积的结果。

文件格式描述可以包含多个类型定义，和一个字段定义。字段定义必须放到所有类型定义的最后。

## 2. 内置类型

1. `int$bits`: 整数类型，其中`$bits`表示其所占的比特数。`$bits`为空时表示`int32`
2. `char$bits`: 字符类型，其中`$bits`表示其所占的比特数。`$bits`为空时表示`char8`
3. `float$bits`: 浮点类型，其中`$bits`表示其所占的比特数。`$bits`为空时表示`float32`
4. `byte`: 字节类型

其中`float32`和`float64`采用`IEEE 754`标准。

## 3. 内置函数

1. `prod` 伪代码如下：
```
function prod(array) begin
    value = 1
    for (item in array) value *= item
    return value
end function
```

2. `sum` 伪代码如下：
```
function sum(array) begin
    value = 0
    for (item in array) value += item
    return value
end function
```

3. `type_bytes` 类`C++`代码如下：
```cpp
int type_bytes(DTYPE dtype) {
    switch (dtype) {
        case VOID: return 0;
        case INT8: return 1;
        case UINT8: return 1;
        case INT16: return 2;
        case UINT16: return 2;
        case INT32: return 4;
        case UINT32: return 4;
        case INT64: return 8;
        case UINT64: return 8;
        case FLOAT16: return 2;
        case FLOAT32: return 4;
        case FLOAT64: return 6;
        case PTR: return sizeof(void*);
        case CHAR8: return 1;
        case CHAR16: return 2;
        case CHAR32: return 4;
        case UNKNOWN8: return 1;
        case UNKNOWN16: return 2;
        case UNKNOWN32: return 4;
        case UNKNOWN64: return 8;
        case UNKNOWN128: return 16;
    }
    return 0;
}
```

这里需要说明的是，`type_bytes`是用来计算某个类型表示的长度的。  
内置支持的类型的枚举声明如下：
```cpp
enum DTYPE {
    VOID        = 0,
    INT8        = 1,
    UINT8       = 2,
    INT16       = 3,
    UINT16      = 4,
    INT32       = 5,
    UINT32      = 6,
    INT64       = 7,
    UINT64      = 8,
    FLOAT16     = 9,
    FLOAT32     = 10,
    FLOAT64     = 11,
    PTR         = 12,              ///< for ptr type, with length of sizeof(void*) bytes
    CHAR8       = 13,            ///< for char saving string
    CHAR16      = 14,           ///< for char saving utf-16 string
    CHAR32      = 15,           ///< for char saving utf-32 string
    UNKNOWN8    = 16,        ///< for self define type, with length of 1 byte
    UNKNOWN16   = 17,
    UNKNOWN32   = 18,
    UNKNOWN64   = 19,
    UNKNOWN128  = 20,

    BOOLEAN     = 21,    // bool type, using byte in native
    COMPLEX32   = 22,  // complex 32(16 + 16)
    COMPLEX64   = 23,  // complex 64(32 + 32)
    COMPLEX128  = 24,  // complex 128(64 + 64)
};

```


## 4. 模块存储格式

模块存储格式声明：

```
string := <int32:size><char8*$size:data>;
prototype := <int8:dtype><int32:dims><int32*dims:shape>;
tensor := <prototype:proto>
    <byte*(prod($proto.shape) * type_bytes($proto.dtype)):memory>;
packed_tensor := <int32:size><tensor*$size:fields>;
bubble := <int32:size><[<string:name><packed_tensor:value>]*$size:params>
inputs := <int32:size><int32*$size:indexs>;
node := <bubble:bubble><inputs:inputs>;
graph := <int32:size><node*$size:nodes>;
outputs := <int32:size><int32*$size:indexs>;
module := <inputs:inputs><outputs:outputs><graph:graph>;
header := <int32:fake><int32:code><byte*120:data>;
<header><module>
```

----
其中 `$tensor.memory` 存放的是 `$tensor.proto.dtype` 描述的类型号的内存。
关于类型号和类型以及对应关系的描述见`3.3`节。

`$node.inputs` 表示对应 `$node` 节点的输入节点。
节点用整数索引表示，索引值为该节点在 `$graph.nodes` 中的下标。

`$module.inputs` 和 `$module.outputs` 分别表示输入节点和输出节点在 `$module.graph.nodes` 中的下标值。

`$header.code` 为版本标识符，当前支持的版本为：
格式V1为`0x19910929`。

`$header.fake` 为保留字节，无实际意义。

`$header.data` 为保留字节区，长度为120个字节。保留字节可用于用户自定义。

`$bubble.params.name` 的长度在 `[0, 32)`。