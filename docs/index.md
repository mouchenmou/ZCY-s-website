# Welcome to ZCY's World

*测试实时重载功能 - 2025年11月30日 13:53*

<div class="hero-banner">
  <div class="profile-card">
    <div class="avatar-container">
      <img src="assets/IMG/1.jpg" alt="某晨某" class="profile-avatar">
    </div>
    <h2 class="profile-name">某晨某</h2>
    <div class="social-links">
      <a href="https://github.com/mouchenmou" target="_blank" class="social-link github" title="GitHub">
        <i class="fab fa-github"></i>
      </a>
      <a href="mailto:2787014694@qq.com" class="social-link email" title="Email">
        <i class="fas fa-envelope"></i>
      </a>
      <a href="https://www.douyin.com/user/MS4wLjABAAAADdsfimTVarbFoML6kRo8L5zvU96ZGfLLTQfmzCi9R4U?from_tab_name=main" target="_blank" class="social-link douyin" title="抖音">
        <i class="fab fa-tiktok"></i>
      </a>
      <a href="https://music.163.com/#/user/home?id=3871698908" target="_blank" class="social-link music" title="网易云音乐">
        <i class="fas fa-music"></i>
      </a>
    </div>
    <p class="profile-quote">
      "你路过我的世界，留下短暂的回忆，我却用你装饰了我的整个青春。"
    </p>
  </div>
</div>

<div class="website-stats">
  <div class="running-time">
    <i class="fas fa-calendar-alt"></i>
    <div class="date-info">
      <span class="current-date">今天是 <span id="current-date"></span></span>
      <span class="current-weekday">星期<span id="current-weekday"></span></span>
    </div>
    <div class="divider"></div>
    <span class="stats-label">此网站已运行</span>
    <div class="time-display">
      <span id="days">0</span> 天 
      <span id="hours">0</span> 小时 
      <span id="minutes">0</span> 分钟 
      <span id="seconds">0</span> 秒
    </div>
  </div>
</div>

<script>
// 设置网站开始日期（您可以修改这个日期）
const websiteStartDate = new Date('2025-09-13 00:00:00');

function updateDateTime() {
  const now = new Date();
  
  // 更新当前日期
  const year = now.getFullYear();
  const month = now.getMonth() + 1;
  const date = now.getDate();
  document.getElementById('current-date').textContent = `${year}年${month}月${date}日`;
  
  // 更新星期
  const weekdays = ['日', '一', '二', '三', '四', '五', '六'];
  const weekday = weekdays[now.getDay()];
  document.getElementById('current-weekday').textContent = weekday;
  
  // 更新网站运行时间
  const timeDiff = now - websiteStartDate;
  
  const days = Math.floor(timeDiff / (1000 * 60 * 60 * 24));
  const hours = Math.floor((timeDiff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((timeDiff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((timeDiff % (1000 * 60)) / 1000);
  
  document.getElementById('days').textContent = days;
  document.getElementById('hours').textContent = hours;
  document.getElementById('minutes').textContent = minutes;
  document.getElementById('seconds').textContent = seconds;
}

// 立即执行一次，然后每秒更新
updateDateTime();
setInterval(updateDateTime, 1000);
</script>


## 欢迎来到ZCY的小世界

<div class="personal-description">
  <!-- 这里将来放个人描述内容 -->
  <p class="placeholder-text">大家好，这里是一个小彩笔的blog，用于记录我的一些课程笔记还有实验室的一些内容，大家如果有需要的也可以参阅，感谢支持。</p>
</div>
