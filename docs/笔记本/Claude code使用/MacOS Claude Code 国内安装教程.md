### 1. 安装环境：

```bash
sudo apt install nodejs npm
```

### 2. 使用国内镜像安装 claude：

```bash
npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com
```

### 3. 运行 claude code：

```bash
claude
```

这时候会提示 `Unable to connect to Anthropic services`，不要慌，继续下一步。

### 4. 打开配置文件：

```bash
cd ~
vi .claude.json
```

### 5. 按一下 `i` 进入编辑模式，然后在大括号里加一句：（注意和其它字段对齐）

```json
"hasCompletedOnboarding": true,
```

然后按一下 `Esc`，输入 `:wq`，回车，保存退出。

### 6. 配置自己的API key
[这个教程](阿里云_API_白嫖教程.pdf)能白嫖300块钱的免费额度，速速行动。

### 7. 打开另一个配置文件：

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

!!! explanation
    由于我白嫖了千问的300元免费额度，因此我上面的ANTHROPIC_MODEL写的是`qwen3.6-plus`。如果你想用别的模型，比如deepseek这些，可以参考一下[这个教程](https://www.bilibili.com/video/BV1rBRQBSEwB/?spm_id_from=333.337.search-card.all.click&vd_source=d6e21125b4ff2bde83430369b29c9418)，这个up主直接在ccswitch里面进行配置，会方便很多。但是我觉得经历一下上面的配置过程能加深对claude code的理解，因此建议初学者先跟着上面的步骤走一遍，之后再用ccswitch。

### 8. 现在再运行

```bash
claude
```

你应该能看到一个小宠物，下面显示 ` qwen3.6-plus · API Usage Billing `，然后就可以愉快使用了！