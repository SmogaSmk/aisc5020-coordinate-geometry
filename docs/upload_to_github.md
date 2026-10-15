# 上传到 GitHub

本地仓库已经有 `main` 分支和提交历史；当前还没有创建或推送远端仓库。

1. 登录 GitHub，打开 https://github.com/new 。
2. 创建 `aisc5020-coordinate-geometry`。若希望组员直接查看和 fork，选择 Public。不要勾选自动生成 README、gitignore 或 license，本地已有项目文件。
3. 在这个项目目录运行下方命令，把 `YOUR_USERNAME` 换成自己登录的 GitHub 用户名：

```powershell
git remote add origin https://github.com/YOUR_USERNAME/aisc5020-coordinate-geometry.git
git push -u origin main
```

Git 若弹出登录或授权，按 GitHub 的流程完成。上传完成后确认网页能看到 README、src、tests 和 docs，再把网页链接发群。

如果从 ZIP 恢复项目，解压后先检查 `git status`：

- 若不是 Git 仓库，可运行 `git init -b main`，然后 `git add .`、`git commit -m "Initialize shared project and Module 5"`，再执行上面的远端命令。
- 配套 `.bundle` 保留现有提交历史，可用 `git clone <bundle文件路径> aisc5020-coordinate-geometry` 恢复源码仓库。运行结果另在交付 ZIP 中。

第一次推送后，GitHub Actions 会自动运行数学测试、smoke 和第 5 部分验证。此处提供配置，不代表远端检查已经执行成功。

组员无需等待写权限，可以 fork 后在自己的分支开发并提交 PR。需要直接在原仓库创建分支时，再由维护者邀请其 GitHub 账号。含学号的 proposal 和原始课程资料保留在公开仓库之外。
