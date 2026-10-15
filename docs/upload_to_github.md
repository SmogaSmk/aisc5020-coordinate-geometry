# GitHub 协作与本地恢复

项目已上传到公开仓库：https://github.com/kvvvvvv23/aisc5020-coordinate-geometry 。`main` 保存可运行版本。

## 组员开始开发

1. 打开上面的仓库，使用自己的账号点击 Fork。
2. 克隆自己的 fork，按 README 建立环境，在自己的分支开发。
3. 提交后向原仓库的 `main` 发起 Pull Request，附运行命令及结果。详细步骤见根目录 `CONTRIBUTING.md`。

只想查看或运行时，可以直接克隆原仓库：

```powershell
git clone https://github.com/kvvvvvv23/aisc5020-coordinate-geometry.git
cd aisc5020-coordinate-geometry
```

报告 PDF 和原始验证结果在仓库的 Releases / `module5-v1` 下。

## 从交付包恢复

如果从 ZIP 恢复项目，解压后先检查 `git status`：

- 日常协作优先从 GitHub clone；单纯解压 ZIP 不包含 `.git` 目录。
- 配套 `.bundle` 保留现有提交历史，可用 `git clone <bundle文件路径> aisc5020-coordinate-geometry` 恢复源码仓库。运行结果另在交付 ZIP 中。

每次推送或 PR 后，GitHub Actions 会运行数学测试、smoke 和第 5 部分验证。原始实现提交 `d616233` 的远端检查已成功；后续提交请查看 Actions 页对应的状态。

组员无需等待写权限，可以 fork 后在自己的分支开发并提交 PR。需要直接在原仓库创建分支时，再由维护者邀请其 GitHub 账号。含学号的 proposal 和原始课程资料保留在公开仓库之外。
