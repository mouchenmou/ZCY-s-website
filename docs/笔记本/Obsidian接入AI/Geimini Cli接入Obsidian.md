
## 1. 安装Gemini Cli

参考这个教程：[Gemini Cli安装教程](https://www.bilibili.com/video/BV1cFKZzcEiV/?spm_id_from=333.1391.0.0&vd_source=d6e21125b4ff2bde83430369b29c9418)

1. 第一步：安装node.js
2. 第二步：在Powershell或者CMD中输入：`npm install -g @google/gemini-cli`
3. 第三步：输入：`gemini`

### 我在登陆的时候遇到了以下几个问题：

#### 1. 网页端登陆失败了：

#### 方法一：新建一个终端，并临时设置代理：

`$env:HTTP_PROXY="http://127.0.0.1:你的端口号"`
`$env:HTTPS_PROXY="http://127.0.0.1:你的端口号"`

#### 方法二：开tun模式，以后每次要用gemini cli就开tun模式。

#### 方法三：一劳永逸
但是，  开启了tun模式之后就没法用claude了：
![](附件/Pasted%20image%2020260506084533.png)
所以在shell 配置文件中设置环境变量（一劳永逸，不用每次手动配）：

##### Windows系统的解决方法：
 如果你用的是 PowerShell（Windows 默认终端），持久化环境变量要在 $PROFILE 文件里设置。在 PowerShell 里运行：`notepad $PROFILE`，然后在文件里加上：
 - $env:HTTPS_PROXY = "http://127.0.0.1:你的端口"
 - $env:HTTP_PROXY = "http://127.0.0.1:你的端口"
  
  如果你用的是 Git Bash（MINGW64），那 ~/.bashrc 的路径是 C:\Users\admin\.bashrc，直接编辑就行。

 
因为我担心加上了那两句话之后，我在终端中又打不开Claude了，所以Claude帮我想了个办法：条件判断方式——只在代理真的在跑的时候才生效：在 $PROFILE 中写下面这段话：

```
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $tcp.Connect("127.0.0.1", 你的端口)
    $tcp.Close()
    $env:HTTPS_PROXY = "http://127.0.0.1:你的端口"
    $env:HTTP_PROXY = "http://127.0.0.1:你的端口"
} catch {}
```

去了美国之后删掉，在美国不用开代理，这个反而会坑你

注意：我第一次运行这个命令的时候，它给我创建了一个新的文件，路径为：C:\Users\admin\Documents\WindowsPowerShell

在执行完上面的操作之后，我还遇到了一个问题：我能够在Powershell打开gemini和claude了，但是我在Obsidian的Terminal插件中还是打不开gemini，所以需要把Terminal插件中的默认终端改成Powershell，需要在前置中选择`Powershell: 整合式`，然后在配置中将其设置为整合式，在默认配置中选中它，就解决了
![](附件/Pasted%20image%2020260506093414.png)


##### Unix系统（MacOS，Linux）的解决方法：

 - macOS（从 Catalina 起）默认 shell 是 zsh → 用 ~/.zshrc
  - Linux 多数默认是 bash → 用 ~/.bashrc
  
在 ~/.bashrc 或 ~/.zshrc 中加入下面两句话：
- export HTTPS_PROXY=http://127.0.0.1:你的代理端口
- export HTTP_PROXY=http://127.0.0.1:你的代理端口

去了美国后得删掉，因为去了美国就没必要开梯子了。
#### 2. 说我年龄未满18岁
我发现虽然我的个人信息里面显示的生日是满了18岁的，但是我还没有认证年龄，所以还得认证一下年龄。

### 3. 教程中提到的users must configure GOOGLE_CLOUD_PROJECT

打开谷歌云，点击控制台，点击My Project新建一个项目，然后记录下这里面的ID，然后再打开C盘，Users，admin找到`.gemini`这个文件夹，新建一个文件命名为`.env`，然后在这个文件里面输入：`GOOGLE_CLOUD_PROJECT = "这里面填刚才记录的ID"`

## 2. 用Terminal插件把Gemini装入obsidian


参考这个教程：[Obsidian党狂喜！用Terminal插件把Gemini 3 Pro装进笔记里啦](https://www.bilibili.com/video/BV1XWrDBPEQ3/?spm_id_from=333.337.search-card.all.click&vd_source=d6e21125b4ff2bde83430369b29c9418)

不过这个视频教程是以MacOS为例讲解的，我是接入到Windows系统里面的

第一步：安装Terminal插件

我在Windows电脑中的Obsidian中安装了Terminal插件之后尚未进行其它任何操作，点击Open Terminal之后显示如下：![](附件/Pasted%20image%2020260505174946.png)

选择win32IntegratedDefault，这是Windows自带的Powershell。我已经在Terminal插件中将默认终端设置成这个终端了。就是每次打开终端都是上下分屏，需要手动改成左右分屏。

装了这个terminal插件之后不管什么AI都能用，Gemini，Claude这些都可以

### MacOS的解决办法：

视频教程演示的是MacOS，MacOS安装完Terminal插件之后，由于它没办法读取系统的那个路径变量，在里面输入了调用gemini的命令之后可能会显示command not found。

在桌面打开一个终端，输入：

1. 打开配置文件：`nano ~/.zshrc` 
2. 粘贴环境变量：`export PATH="$PATH:/opt/homebrew/bin"`
3. 保存并退出：`ctrl + 0`写入保存，`Enter` 确认文件名，`ctrl + X`推出编辑器
4. 立即生效：`source ~/ .zshrc`


