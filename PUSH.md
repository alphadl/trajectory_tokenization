# 推送说明

在本机终端执行：

```bash
cd trajectory_tokenization
chmod +x push.sh
./push.sh
```

若推送失败，按下面排查。

## 1. HTTPS 提示需要登录

- 浏览器会弹窗：用 GitHub 账号登录即可。
- 或使用 Personal Access Token：GitHub → Settings → Developer settings → Personal access tokens 生成 token，推送时「密码」填 token。

## 2. 改用 SSH（已配置 SSH 公钥时）

```bash
git remote set-url origin git@github.com:alphadl/trajectory_tokenization.git
git push -u origin main
```

## 3. 提示 "Permission denied" 或 "Authentication failed"

- 确认本机已登录 GitHub（浏览器或 `gh auth login`）。
- 确认仓库归属为 `alphadl`，且当前账号有 push 权限。
