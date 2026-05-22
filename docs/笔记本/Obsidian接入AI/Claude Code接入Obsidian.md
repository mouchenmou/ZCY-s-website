
!!! warning "不建议在Windows主系统中装Claude Code"
    我原本是将Claude Code安装在Windows的主系统中，但它在Windows下真的很笨，本身就更适配Linux。而且把AI工具装在主系统中会有安全隐患，所以我在WSL中也装了一个Claude Code，顺便把WSL从C盘迁移到了D盘。原本把Windows主系统中的Claude Code直接删掉了，但是Claudian用不了了，只好装回来。
    
    但是在Terminal插件需要把默认的Shell改成WSL的bash，这样太麻烦了，因为Gemini Cli还得在Powershell中对话，所以不如Terminal插件用来跟Gemini Cli对话，Claudian用来跟Claude对话。


视频教程：[不止Claude Code:任意AI接入Obsidian打造本地知识库,Gemini/GLM/Codex/MiniMax全搞定](https://www.bilibili.com/video/BV1BgoVBGEox/?spm_id_from=333.337.search-card.all.click&vd_source=d6e21125b4ff2bde83430369b29c9418)

1. 第一步 配置Claude Code：由于我之前已经配置过了，我就不详细讲了，如果还没配置的可以参考我的[[MacOS Claude Code 国内安装教程|Claude code配置的教程]]。CCswitch的话去官网下载一下很快的。
2. 第二步 安装claudian插件：这个插件在Obsidian中是没有的，点击这个[claudian](https://github.com/YishenTu/claudian)就能跳转了
3. 第三步：在终端中打开Claude code并告诉它：我的某一个路径中有这样的一个vault，帮我把`https://github.com/YishenTu/claudian`这个插件安装进去。

我在配置的时候遇到了加载Error的问题，解决方法如下：

1. 第一步：打开Claudian插件，选中Claude，找到Custom Model并输入你的API的模型
2. 第二步：找到自定义环境变量，把这里面的东西改成：

ANTHROPIC_API_KEY=你的千问API密钥

ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic

ANTHROPIC_MODEL=qwen3.6-plus（这里填你的模型）

注意，那个框框里面的最后一行还有一个CLAUDE_CODE_USE_BEDROCK=1，这一行千万不要添加，这个是claude自己的模型才添加的。
![](附件/Pasted%20image%2020260506082409.png)


不过Terminal插件已经能够让我在Obsidian中使用claude了，但是在Windows系统中，Terminal插件的终端渲染出来的效果真的很差，看着非常难受，在MacOS中看着就会舒服很多，很乱，所以我在Windows中用Claudian会舒服很多，并且跟Gemini Cli的终端不冲突。在MacOS中我直接用Terminal插件跟claude code对话。


---

### 让 Claude Code 干活

刚把Claude Code接入到WSL后，我就让它帮我梳理了一下我的Ubuntu，感觉还是蛮不错的：

![](附件/Pasted%20image%2020260508174657.png)