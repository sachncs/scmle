// Regenerates the committed raster assets (social image, touch icons) from
// SVG/HTML using the Playwright Chromium that the e2e tests already need.
// Run manually: node scripts/make-assets.mjs
import { readFile, writeFile } from "node:fs/promises";
import { chromium } from "@playwright/test";

const publicDir = new URL("../public/", import.meta.url);
const mark = await readFile(new URL("scmle-mark.svg", publicDir), "utf8");
const markUri = `data:image/svg+xml;base64,${Buffer.from(mark).toString("base64")}`;

const og = `<!doctype html><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0}
body{width:1200px;height:630px;background:#0b0b0d;color:#f5f5f7;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",system-ui,"Segoe UI",Roboto,sans-serif;position:relative;overflow:hidden}
.glow{position:absolute;inset:-200px -100px auto -100px;height:700px;background:radial-gradient(50% 50% at 50% 0%,rgba(69,217,160,.22),transparent 70%)}
.wrap{position:relative;padding:72px 80px;height:100%;display:flex;flex-direction:column}
.brand{display:flex;align-items:center;gap:18px;font-size:34px;font-weight:650;letter-spacing:.02em}
.brand img{width:56px;height:56px}
h1{margin-top:auto;font-size:104px;line-height:1.02;letter-spacing:-.045em;font-weight:700;max-width:980px}
h1 span{color:#9a9aa1}
p{margin-top:28px;font-size:34px;line-height:1.35;color:#b4b4ba;max-width:900px}
.foot{margin-top:40px;font-size:26px;color:#45d9a0;font-weight:600}
</style><div class="glow"></div><div class="wrap"><div class="brand"><img src="${markUri}">SCMLE</div>
<h1>Mathematics <span>meets engineering.</span></h1>
<p>A curation-first machine learning lifecycle with leakage-safe evaluation and signed artifacts.</p>
<div class="foot">sachncs.github.io/scmle</div></div>`;

const icon = (size, radius) => `<!doctype html><meta charset="utf-8"><style>*{margin:0}body{width:${size}px;height:${size}px;background:transparent}img{width:${size}px;height:${size}px;display:block}</style><img src="${markUri}">`;

const browser = await chromium.launch();
async function shot(html, w, h, file, omitBackground = false) {
  const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
  await page.setContent(html);
  await writeFile(new URL(file, publicDir), await page.screenshot({ omitBackground }));
  await page.close();
}
await shot(og, 1200, 630, "og.png");
await shot(icon(512), 512, 512, "icon-512.png", true);
await shot(icon(192), 192, 192, "icon-192.png", true);
await shot(icon(32), 32, 32, "favicon-32.png", true);
// Apple touch icons are opaque and full-bleed: the OS applies its own mask.
await shot(`<!doctype html><meta charset="utf-8"><style>*{margin:0}body{width:180px;height:180px;background:#111113;display:grid;place-items:center}img{width:124px;height:124px}</style><img src="${markUri}">`, 180, 180, "apple-touch-icon.png");
await browser.close();
console.log("assets written to public/");
