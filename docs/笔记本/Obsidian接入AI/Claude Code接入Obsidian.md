视频教程：[不止Claude Code:任意AI接入Obsidian打造本地知识库,Gemini/GLM/Codex/MiniMax全搞定](https://www.bilibili.com/video/BV1BgoVBGEox/?spm_id_from=333.337.search-card.all.click&vd_source=d6e21125b4ff2bde83430369b29c9418)

1. 第一步 配置Claude Code：由于我之前已经配置过了，我就不详细讲了，如果还没配置的可以参考我的[[Claude code使用/MacOS claude 国内安装教程|Claude code配置的教程]]。CCswitch的话去官网下载一下很快的。
2. 第二步 安装claudian插件：这个插件在Obsidian中是没有的，点击这个[claudian](https://github.com/YishenTu/claudian)就能跳转了
3. 第三步：在终端中打开Claude code并告诉它：我的某一个路径中有这样的一个vault，帮我把`https://github.com/YishenTu/claudian`这个插件安装进去。

我在配置的时候遇到了加载Error的问题，解决方法如下：

1. 第一步：打开Claudian插件，选中Claude，找到Custom Model并输入你的API的模型
2. 第二步：找到自定义环境变量，把这里面的东西改成：

ANTHROPIC_API_KEY=你的千问API密钥

ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic

ANTHROPIC_MODEL=qwen3.6-plus

注意，那个框框里面的最后一行还有一个CLAUDE_CODE_USE_BEDROCK=1，这一行千万不要添加，这个是claude自己的模型才添加的。
![](附件/Pasted%20image%2020260506082409.png)


不过Terminal插件已经能够让我在Obsidian中使用claude了，但是那个终端看着很奇怪，很乱，所以用Claudian会舒服很多。

但是我还遇到了一个问题