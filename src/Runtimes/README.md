## Runtimes

### 说明
- ViewFaceCore.runtime.* 项目是 ViewFaceCore 的运行时依赖项目
- 使用 `ViewFaceCore.Runtimes.sln` 组织管理

### 打包方式
- 运行 `pack.all.bat` 脚本将 Runtimes 下的所有项目打包至 `[输出]\[配置]\[版本号]` 目录下
- 可以通过修改 `pack.all.bat` 脚本中的 `configuration` `version` `output` 参数来修改生成的配置, 版本号和输出