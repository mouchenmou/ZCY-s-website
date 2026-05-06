这个安装就是最无脑的，想要安装桌面端的话直接去官网上安装，想要安装终端版的话就让claude帮你安装。这里我主要是想讲一个问题：

我们会发现在桌面端使用Codex速度真的非常非常慢，不停的reconnecting，解决方法如下：

### Windows 用户
1. 进入：C:\Users\你的用户名\.codex
2. 新建一个文件：.env（注意不是 .env.txt）
3. 粘贴这两行：
    - HTTP_PROXY="http://127.0.0.1:端口号"
    - HTTPS_PROXY="http://127.0.0.1:端口号"
4. 保存后重启 Codex
### macOS 用户
1. 进入：/Users/你的用户名/.codex
2. 新建 .env
3. 粘贴同样内容：
    - HTTP_PROXY="http://127.0.0.1:端口号"
    - HTTPS_PROXY="http://127.0.0.1:端口号"
4. 保存后重启 Codex

去了美国之后得改回来！！！