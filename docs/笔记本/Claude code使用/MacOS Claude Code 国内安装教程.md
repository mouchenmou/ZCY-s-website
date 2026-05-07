安装环境：

```bash
sudo apt install nodejs npm
```

使用国内镜像安装 claude：

```bash
npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com
```

运行 claude code：

```bash
claude
```

这时候会提示 `Unable to connect to Anthropic services`，不要慌，继续下一步。

打开配置文件：

```bash
cd ~
vi .claude.json
```

按一下 `i` 进入编辑模式，然后在大括号里加一句：（注意和其它字段对齐）

```json
"hasCompletedOnboarding": true,
```

然后按一下 `Esc`，输入 `:wq`，回车，保存退出。

打开另一个配置文件：

```bash
cd ~/.claude/
vi settings.json
```

按一下 `i` 进入编辑模式，然后把下面的内容加进去：

```json
{
  "env": {
	  "ANTHROPIC_AUTH_TOKEN": "sk-***",
	  "ANTHROPIC_BASE_URL": "https://dashscope.aliyuncs.com/apps/anthropic",
	  "ANTHROPIC_MODEL": "qwen3.6-plus"
  }
}
```

这里 `sk-***` 替换为你自己的 API-KEY。然后按一下 `Esc`，输入 `:wq`，回车，保存退出。

现在再运行

```bash
claude
```

你应该能看到一个小宠物，下面显示 ` qwen3.6-plus · API Usage Billing `，然后就可以愉快使用了！