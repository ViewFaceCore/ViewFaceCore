# OpenRoleZoo

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

原 OpenRoleCraft 项目搁置，换名 OpenRoleZoo 启动。

## 文件结构

* `include`：头文件路径，`include/orz` 下的头头文件会安装到 `include` 目录下。
* `src`：源文件路径，不想被安装的头文件也可以放到这里。
* `test`：测试程序目录，每个单独源文件都会编译成测试程序。
* `tools`：工具目录，每个单独源文件都会编译成可执行程序，最后安装到 `bin` 目录下。

## 源代码扩展

将需要添加的源文件放入对应的文件目录下，重新执行 `cmake` 并编译即可。

## 编译安装

运行 craft 中对应的脚本，即可将对应的库安装到系统中。

## 其他说明

本项目开发平台为 `windows` + `CLion` + `MSYS2`。

## 联系我们

`e-mail`: `likier@sina.cn`
