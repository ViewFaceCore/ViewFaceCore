# TenniS

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

<div align="divcss5">
<img src="./logo/TenniS-H.png" width="640"/>
</div>

> TenniS: Tensor based Edge Neural Network Inference System 

## Compilation

The default installation DIR is ` ${PROJECT_BINARY_DIR}/../build`,
add `-DCMAKE_INSTALL_PREFIX` to change installation DIR.

1. Do `cmake` or `CPU`:
```
cmake ..
-DTS_USE_OPENMP=ON
-DTS_USE_SIMD=ON
-DTS_ON_HASWELL=ON
-DTS_BUILD_TEST=OFF
-DTS_BUILD_TOOLS=OFF
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=/usr/local
```
or `GPU`:
```
cmake ..
-DTS_USE_CUDA=ON
-DTS_USE_CUBLAS=ON
-DTS_USE_OPENMP=ON
-DTS_USE_SIMD=ON
-DTS_ON_HASWELL=ON
-DTS_BUILD_TEST=OFF
-DTS_BUILD_TOOLS=OFF
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX=/usr/local
```

> Option `TS_ON_HASWELL` means support `AVX2` and `FMA`.  
> Option `TS_ON_SANDYBRIDGE` means only support `AVX2` but no `FMA`.  
> Option `TS_ON_PENTIUM` means only support `SSE2`.  

If want compile all instructions support, switch `TS_DYNAMIC_INSTRUCTION` ON.
Notice: `TS_DYNAMIC_INSTRUCTION` ONLY work in release version.

~~When compilation target has no instruction-set like `AVX` or `FMA`,
**MUST** turn corresponding option off.~~

When compilation target is `arm-*`, **PLEASE** set `-DTS_ON_ARM=ON`.
When compilation target is `arm-v7`, **MUST** set `-DTS_ON_ARMV7=ON`.

2. Do `make -j16` and waiting.

3. Do `make install`.

4. Find headers and libraries in `CMAKE_INSTALL_PREFIX`
