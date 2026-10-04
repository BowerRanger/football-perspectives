// Seek the dashboard's /api/video/<shot> in Chromium to several currentTime targets and
// dump the displayed frame as PNG, to compare against cv2's exact decode.
const { chromium } = require('playwright');
const fs = require('fs');
const OUT = '/Users/joebower/.claude/jobs/88133eb1/tmp/look/seek';
(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ channel: 'chrome' });
  const page = await browser.newPage();
  await page.goto('http://localhost:8780/');
  const fps = 30;
  const targets = [];
  for (const f of [414, 415, 416, 417]) {
    targets.push({ name: `exact_${f}`, t: f / fps });
    targets.push({ name: `mid_${f}`, t: (f + 0.5) / fps });
  }
  const res = await page.evaluate(async ({ targets }) => {
    const v = document.createElement('video');
    v.muted = true; v.preload = 'auto'; v.src = '/api/video/origi01';
    document.body.appendChild(v);
    await new Promise((r) => v.addEventListener('loadeddata', r, { once: true }));
    const c = document.createElement('canvas'); c.width = 1920; c.height = 1080;
    const ctx = c.getContext('2d');
    const out = {};
    for (const { name, t } of targets) {
      v.currentTime = t;
      await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
      await new Promise((r) => setTimeout(r, 250));
      ctx.drawImage(v, 0, 0, 1920, 1080);
      out[name] = c.toDataURL('image/png');
    }
    return out;
  }, { targets });
  for (const [k, d] of Object.entries(res)) fs.writeFileSync(`${OUT}/${k}.png`, Buffer.from(d.split(',')[1], 'base64'));
  await browser.close();
  console.log('ok', Object.keys(res).length);
})();
