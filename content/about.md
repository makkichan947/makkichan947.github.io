+++
date = '2025-10-01T12:40:57+08:00'
draft = false
title = 'About'
comments = true
+++


```
███╗   ███╗ █████╗ ██╗  ██╗██╗  ██╗██╗ ██████╗██╗  ██╗ █████╗ ███╗   ██╗
████╗ ████║██╔══██╗██║ ██╔╝██║ ██╔╝██║██╔════╝██║  ██║██╔══██╗████╗  ██║
██╔████╔██║███████║█████╔╝ █████╔╝ ██║██║     ███████║███████║██╔██╗ ██║
██║╚██╔╝██║██╔══██║██╔═██╗ ██╔═██╗ ██║██║     ██╔══██║██╔══██║██║╚██╗██║
██║ ╚═╝ ██║██║  ██║██║  ██╗██║  ██╗██║╚██████╗██║  ██║██║  ██║██║ ╚████║
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
                                                                         
```

# 你好
我是Yaku Makki，你可以管我叫喵帕

一个普通的初三生，以及5年编程经验（自认为）的全栈开发者（对，对吗？）

# 成分
成分非常地复杂，多到连自己也认不出是个什么东西了：
密码学，自制系统，Python，Rust，C/C++，LLM；
BA牢玩家，MC生电人；40K牢兵，安东星超人；罪大恶极的P社战犯，又菜又爱玩的音游人；
（写不下了）

# GitHub状态

<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=makkichan947&show_icons=true&theme=radical" height="160"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=makkichan947&layout=compact&theme=radical" height="160"/>
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=makkichan947&theme=github-compact" />
</p>

# 联系方式
- QQ: 3521866332
- Email: [yakumakki947@hotmail.com](mailto:yakumakki947@hotmail.com)/[nekosparry0727@outlook.com](mailto:nekosparry0727@outlook.com)
- GitHub: [RealMakkichan](https://github.com/makkichan947)
- Twitter: [Makki_Yaku947](https://x.com/Makki_Yaku947)

# 平台
- BiliBili :[喵帕斯卡利](https://space.bilibili.com/3546566522571572)
- Twitch: [makkichan947](https://www.twitch.tv)

# 关于本站
本站本来计划使用Hugo M10C，因为想自己写所以自己整了一个

曾使用HTML+CSS+JS来搭建，详情见[/post/starthistory.md](./post/starthistory.md)

你可以通过输入密码来解锁下面的隐藏区域:)

<!-- ===== 动态解锁开始 ===== -->
<!-- ===== PASSWD:270711 ===== -->
<div id="hidden" style="display:none;"></div>

<div id="locker" style="margin-top:2em;">
  <label>
    <input id="secretInput" type="password" placeholder="输入暗号解锁隐藏信息" style="padding:4px 8px;">
    <button onclick="unlock()">解锁</button>
  </label>
  <p id="wrongHint" style="color:#d33;display:none;">请输入正确的暗号:/</p>
</div>

<script type="module">
const HASH = '0534f0cc06348c0b8a52e46bb373656f498a1cba3330d7b888256743ab1c55ad';
const NI_GE_XIAO_REN_yougotme = '/about_secret/';

async function sha256(str) {
  const buf = new TextEncoder().encode(str);
  const hash = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(hash))
              .map(b => b.toString(16).padStart(2, '0'))
              .join('');
}

window.unlock = async function () {
  const val = document.getElementById('secretInput').value;
  if (await sha256(val) !== HASH) {
    document.getElementById('wrongHint').style.display = 'inline';
    return;
  }
  // 拉取隐藏 Markdown → 转成 HTML → 填进去
  const resp = await fetch(NI_GE_XIAO_REN_yougotme);
  const mdText = await resp.text();
  // 这里用最简 Markdown 解析器 marked（2.5 kB gzip）
  const { marked } = await import('https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js');
  document.getElementById('hidden').innerHTML = marked.parse(mdText);
  document.getElementById('hidden').style.display = 'block';
  document.getElementById('locker').style.display = 'none';
};
</script>
<!-- ===== 动态解锁结束 ===== -->
