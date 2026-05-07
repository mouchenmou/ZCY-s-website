我原本是把Claude Code安装在Windows原生系统下面的，但是在原生系统下面真的很蠢。。。Claude Code本身就更适配Linux系统。而且把Claude装在主系统中会很危险，搞不好哪天把你电脑搞死了，所以我选择跑路。。。

首先打开Ubuntu，刚打开的时候是这样的：
![](附件/Pasted%20image%2020260507215446.png)
提示中提到的“检测到 localhost 代理配置”是一个非常经典的 **WSL 网络同步问题**。因为我的 Windows 开了代理，但 WSL 默认是一个独立的虚拟网络，它感知不到 Windows 上的代理，这会导致后续 `npm install` 报错或者 `claude` 无法联网。

为了让 WSL 顺畅地访问网络，最简单的办法是开启 WSL 的“镜像网络”模式：

1. 在 Windows 中按下 `Win + R`，输入 `%USERPROFILE%` 并回车。
2. 在弹出的文件夹里找一下有没有 `.wslconfig` 文件。如果没有，就**新建一个文本文件**，命名为 `.wslconfig`（注意前面有个点）。
3. 用记事本打开它，把下面这段内容贴进去并保存：

```
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

5. 保存后，回到 Windows 终端（PowerShell），输入 `wsl --shutdown` 关闭所有 WSL 窗口。
6. 重新打开你的 Ubuntu。那个“检测到 localhost 代理”的警告应该就消失了，此时 WSL 里的网络会和 Windows 完全同步。

### 第一步：将Ubuntu迁移到D盘
我的C盘只剩30个G能用了，所以要把Ubuntu迁移到D盘，如果你的C盘够用的话其实不迁移也没关系。

在Ubuntu中输入`lsb_release -a`这条命令查看一下Ubuntu系统的版本，我的是24.04。

下面开始我们的操作：

1. 先进入到Powershell中输入 `wsl -l -v` 检查一下Ubuntu是否处于running状态，如果处于running状态的话需要输入`wsl --shutdown`来结束Ubuntu的running状态。
2. 在D盘中创建一个文件夹用于存放WSL，我起的名字是WSL-Ubuntu。
3. 接着在Powershell中输入`wsl --export Ubuntu D:\WSL-Ubuntu\Ubuntu.tar`。
4. 传输完毕后，确认完D盘中存在Ubuntu.tar后，就可以注销原有的wsl了，具体操作为：在终端中输入`wsl --unregister Ubuntu`。
5. 下一步是将备份文件恢复到D盘的WSL-Ubuntu中，在终端中输入`wsl --import Ubuntu D:\WSL-Ubuntu D:\WSL-Ubuntu\Ubuntu.tar`。导入完成之后就可以把`Ubuntn.tar`删掉了，实在是太占内存了。
6. 第5步操作完成之后，在终端中输入`wsl`，会弹出类似`@root...`不啦不啦的东西，我们需要输入`exit`先退出这个根目录。
7. 在终端中输入`Ubuntu config --default-user 你的Linux用户名`。但是这个步骤大多时候会报错，就像我也报错了，我的终端出现了：Ubuntu : 无法将“Ubuntu”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。请检查名称的拼写，如果包括路径，请确保路径正确，然后再试一次。所在位置 行:1 字符: 1...
8. 如果第7步出错了，就在提示符 `#` 后面执行`vi /etc/wsl.conf`，然后点击 `i` 进入编辑模式，切换到文字的最后一行，在最后一行后面输入下面这两行：
```
[user]
default=你的Linux用户名
```
9. 输入完成之后按ESC，输入冒号和保存指令`:wq`，按下enter

#### 我踩到了一个大坑
我在执行完第5步之后再次输入wsl，发现我进入的不是Ubuntu而是dockers-desktop。这是因为docker-desktop变成了默认启动项，因此必须执行`wsl --set-default Ubuntu`将默认启动项改为Ubuntu。

### ==注意==：
我原本的Ubuntu默认处于base环境中，我现在把这个默认给去掉了，以后想进入base这个虚拟环境中就输入`conda activate`（默认进入base），想要退出这个环境就输入`conda deactivate`。如果我想要进入别的虚拟环境，就要输入`conda activate ***`（`***`是这个虚拟环境的名字）

---
### 第二步：安装ccswitch
之前的ccswitch也是安装在主系统中的，所以还得重新在wsl中安装一个，跟着GitHub上的Ubuntu的教程走就行了
![](附件/Pasted%20image%2020260507233940.png)

1. 这边Ubuntu推荐.deb，那我们就找到最新版的.deb文件，复制链接地址之后，在Ubuntu中输入：`wget https://github.com/farion1231/cc-switch/releases/download/v3.14.1/CC-Switch-v3.14.1-Linux-x86_64.deb`（这个链接地址是目前最新版本的，你们不要照搬我的，要是更新了就去复制最新的）。

这一步我又报错了，因为我的Ubuntu是桌面快捷方式，我在创建快捷方式的时候填写的对象位置是`wsl.exe -d Ubuntu`。这时候 Windows 会默认把 **快捷方式文件所在的目录**（也就是你的桌面，对应的路径通常是 `C:\Users\admin\Desktop`）作为起始路径带进 Linux 里。因此我们打开这个桌面快捷方式的属性，然后目标那一栏把`wsl.exe -d Ubuntu`改为`wsl.exe -d Ubuntu --cd ~`就能一劳永逸了，以后打开Ubuntu后就是Linux内部的家目录。

2. 下载完成之后在终端中输入`sudo apt install ./CC-Switch-v3.14.1-Linux-x86_64.deb`（版本一定要填你下载的那个版本）
3. 安装完成之后，在终端输入`cc-switch`就能够弹出ccswich的UI了，但是我的UI中很多字体都是框框，是因为wslg字体问题。只需要在 Ubuntu 终端里执行下面这一行命令，把常用的中文字体补上：


```
sudo apt update && sudo apt install -y fonts-wqy-microhei fonts-wqy-zenhei xfonts-intl-chinese
```

- **`fonts-wqy-microhei`**：文泉驿微米黑，这是 Linux 下非常经典的清晰中文字体。
- **`fonts-wqy-zenhei`**：文泉驿正黑。

---
### 第三步：安装git和Node.js

git我之前已经安装过了，这个去搜一下也很快。

#### 安装Node.js：
1. 下载并运行安装脚本：`curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash`
2. 让Ubuntu立即识别NVM命令：`source ~/.bashrc`
3. 安装最新版本的Node.js：`nvm install node`
4. 安装claude code：`npm install -g @anthropic-ai/claude-code`（运行完这一步之后就不用看下面那个安装Claude Code了）


---
### 第四步：安装Claude Code
[这个地址](https://code.claude.com/docs/en/setup)有非常详细的安装指南。如果不执行`npm install -g @anthropic-ai/claude-code`的话，可以根据这个来。

1. 在终端中输入`curl -fsSL https://claude.ai/install.sh | bash`。安装完成之后会显示：
![](附件/Pasted%20image%2020260508003352.png)
2. 在终端中输入`vim ~/.claude.json`，进入.claude.json文件，然后在键盘中点击i进入编辑模式，在这段文字中加入`"hasCompletedOnboarding": true,`（一定要加上逗号），然后点击ESC退出，输入`:wq`退出编辑，点击enter。
