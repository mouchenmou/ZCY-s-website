!!! warning "不建议在Windows主系统中装Claude Code"
    我原本是将Claude Code安装在Windows的主系统中，但它在Windows下真的很笨，本身就更适配Linux。而且把AI工具装在主系统中会有安全隐患，所以我选择把它迁到WSL中，顺便把WSL从C盘迁移到了D盘。
    
    但是，值得庆祝的是，Claudian插件不需要重新配置原因是 Claudian 插件不走本地 Windows 的 claude 可执行文件，它是通过环境变量直接把 API 请求发到云端：ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic  配置中 cliPath 是空的（"cliPath": ""），说明它走的是 API 直连模式，不需要本地有 claude 命令。
    
    但是在Terminal插件需要把默认的Shell改成WSL的bash，这样太麻烦了，因为Gemini Cli还得在Powershell中对话，所以不如Terminal插件用来跟Gemini Cli对话，Claudian用来跟Claude对话。

  所以即使你删掉了 Windows 上的 Claude Code，Claudian 插件仍然正常工作，它直接用你配置的环境变量连 Dashscope API。

视频教程：[不止Claude Code:任意AI接入Obsidian打造本地知识库,Gemini/GLM/Codex/MiniMax全搞定](https://www.bilibili.com/video/BV1BgoVBGEox/?spm_id_from=333.337.search-card.all.click&vd_source=d6e21125b4ff2bde83430369b29c9418)

1. 第一步 配置Claude Code：由于我之前已经配置过了，我就不详细讲了，如果还没配置的可以参考我的[[MacOS Claude Code 国内安装教程|Claude code配置的教程]]。CCswitch的话去官网下载一下很快的。
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

---
