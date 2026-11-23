# 退款报告交付实验

- `prototype/round-1.html` 和 `round-2.html` 是临时 `prototype` 分支的两轮快照，故意保留归档状态可以直接改变这一原型局限。
- `engineered/index.html` 是增强版，可直接打开。输入约定和审批状态见下方。
- 所有输入均为合成数据。归档仅增加当前页面内的计数。

## 验收约定

1. CSV 首行是 `refund`，后面每行填一个最多两位小数的非负金额。空值和 `NA` 不算，数字 `0` 是有效金额。非法行显示行号，输入上限为 4096 字符。
2. 输入为 `4, NA, 6` 时输出 2 条、均值 5.00。没有有效记录时均值为空，也无法预览归档。
3. 预览只读。操作人确认当前草稿版本后才能归档。取消、修改输入和重复提交均不得产生额外回执。
4. 人工接管用测试脚本分别扮演自动化与审批者。正式服务还需要可信身份和服务端授权。

## 复跑

需要 Node.js 与 Google Chrome。给脚本传入已安装的 Playwright 入口，可选第三个参数为已存在的截图输出目录。页面运行在浏览器临时上下文，使用本地文件。

```bash
node --test experiments/report-delivery/engineered/report-core.test.cjs
lab_dir=$(mktemp -d)
npm install --prefix "$lab_dir" --no-audit --no-fund playwright@1.58.2
node experiments/report-delivery/browser-check.mjs "$lab_dir/node_modules/playwright/index.mjs"
```

`results` 保存两轮原型复核、页面验收与截图。页面载入至首个结果的单次耗时受缓存及机器状态影响，用于示范计时口径。截图坐标是 900×680 视口下的 CSS 像素。
