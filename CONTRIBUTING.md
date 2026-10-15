# 一起开发

每次修改尽量解决一个问题：一个算法、一个推导、一个实验配置或一组分析。先让小例子通过，再扩大实验。

## 没有仓库写入权限：fork + PR

1. 打开仓库，点击 **Fork**，在自己的账号下创建副本。
2. 克隆自己的 fork，进入项目目录。将下方的用户名替换为自己的 GitHub 用户名；仓库名以当前页面显示为准。

   ```powershell
   git clone https://github.com/YOUR_USERNAME/aisc5020-coordinate-geometry.git
   cd aisc5020-coordinate-geometry
   git switch -c role4/preconditioning
   ```

3. 按 README 建立环境并运行测试。完成后只添加本次需要的文件，检查暂存区没有个人信息或本地输出。

   ```powershell
   git status
   git add path/to/changed_file.py
   git diff --cached
   git commit -m "Add diagonal preconditioning validation"
   git push -u origin role4/preconditioning
   ```

4. 回到 GitHub 点击 **Compare & pull request**，确认目标是原项目的 `main`，写明改了什么、如何验证、是否影响公共接口。
5. 按 review 意见继续提交到同一分支，PR 会自动更新。

公开访问与写入权限是两回事。需要直接推送原仓库时，由仓库维护者单独邀请；当前 fork + PR 流程不需要等待邀请。

## 已有写入权限

从最新 `main` 新建自己的分支，通过 PR 合并。不要直接在 `main` 开发，也不要强制推送共享分支。

分支名可以使用 `role1/gd-hb`、`role2/adam`、`role3/runner`、`role4/preconditioning`、`role5/eigenbasis-adam`；同一模块有多人并行时，在后面加具体任务名。

## 合并前检查

- 运行 `python -m unittest discover -s tests -v`。
- 若修改优化器、问题生成或输出格式，另运行 README 中的 smoke 命令。
- 新算法至少用一个可手工推导的小问题验证，不要只看“能运行”。
- 新增实验说明目标、配置、seed、预算、输出含义和失败处理。
- PR 说明中区分已完成验证与尚未运行的主实验，不填入预期数值作为真实结果。
- 不提交学号、账号凭据、`.env`、原始课程 PDF、虚拟环境、缓存或大体积实验输出。

## 公共接口与依赖

模块 3 管理问题生成和批量运行，模块 4 的正式统计依赖其结果。模块 1、2、5 的优化器应遵守统一的 `RunConfig` / `RunResult` 约定。

涉及函数签名、字段名、步数定义、停止规则或结果格式的修改，先开 issue 说明影响，再提交 PR。暂时没有主实验结果时，统计分析可以使用清楚标注的人工小样例开发；不能将这些样例写成项目发现。

## 推导、图和报告

推导写在 `docs/` 下，注明假设和符号；正文中的结论应能对应到证明或实际运行。英文报告段落随本模块一起交付，减少最后整合时重复解释。

图表保留生成脚本与配置。检查坐标、图例、对数轴和失败样本的表示；如果某个算法需要 Hessian、对角线或特征基等额外信息，在图注或实验说明中写清楚。

## 更新自己的 fork

在 GitHub fork 页面点击 **Sync fork** 同步上游 `main`，再在本地切回 `main` 并拉取。保持未提交修改得到保存后再切换分支；有冲突时在 PR 中说明，不要覆盖别人的文件。
