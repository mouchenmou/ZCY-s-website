!!! Warning "真的是踩过的一个大坑"
    我原本已经把Windows主系统中的Claude Code删掉了，因为觉得它在Windows原生系统下面真的很蠢，而且害怕它篡改我的主系统。结果删掉了之后Obsidian中的Claudian打不开了。调教了AI很久最终遗憾立场，不得已把它装回来。但是除了在Obsidian中使用它，其它时候我用的都是WSL中的Claude Code。


## 系统要求


- ✅ **操作系统**：Windows 10 或 Windows 11（**必须是 64 位**）
- ✅ **内存**：建议 4GB 以上
- ✅ **硬盘空间**：约 2-4GB
- ✅ **网络**：需要联网进行安装和使用
- ✅ **权限**：需要管理员权限

---

## 完整安装步骤

### 第一步：安装 Git for Windows以及Node.js（必需）


> ⚠️ **重要**：Claude Code 在 Windows 上依赖 Git Bash 运行，必须先安装 Git。

#### 1.1 下载 Git


访问 Git 官网下载页面：

```
https://git-scm.com/downloads/win
```

点击下载最新的 **64 位版本**。

#### 1.2 安装 Git

1. 双击下载的安装文件（如 `Git-2.51.2-64-bit.exe`）
2. **全程使用默认选项**，一路点击 "Next"
3. 确保在 "Adjusting your PATH environment" 步骤中选择：
    - ✅ **"Git from the command line and also from 3rd-party software"**（默认选项）
4. 点击 "Install" 开始安装
5. 完成后点击 "Finish"

#### 1.3 验证 Git 安装

1. 按 `Win + R`
2. 输入 `cmd`，按回车
3. 在命令提示符中输入：

```shell
git --version
```

✅ 如果显示类似 `git version 2.51.2.windows.1`，说明安装成功。

#### 1.4 安装Node.js
我已经在WSL中配置Claude Code教程中讲过了，这里不讲了

---

### 第二步：安装 Claude Code

#### 2.1 打开 PowerShell


1. 按 `Win` 键
2. 输入 `powershell`
3. **右键**点击 "Windows PowerShell"

#### 2.2 运行安装命令

在 PowerShell 中复制并运行以下命令：

```powershell
npm install -g @anthropic-ai/claude-code
```

#### 2.3 等待安装完成

安装过程会显示进度信息，完成后会显示：

```
added 2 packages in 4s
```

---

### 第三步：安装ccswitch

来到[ccswitch的仓库地址](https://github.com/farion1231/cc-switch)，点击Latest release，往下滑找到Windows版本的安装包（其它系统的就下载自己系统的安装包就好了）。

---

### 第四步：配置API key


选择你自己想要的厂商，我买了Deepseek的，还能去白嫖千问的。