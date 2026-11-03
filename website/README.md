# scmle marketing site

The marketing site is a small static Astro site. It is intentionally kept
separate from the Python package so the product story can evolve without
coupling the runtime distribution to a frontend toolchain.

```bash
npm install
npm run dev
npm run build
```

The site is static output and is built in CI with `npm ci`, a high-severity
production dependency audit, and `npm run build`.

## Design system

* Tokens (colour, type scale, 8-pt spacing, radii) live in `src/styles/tokens.css`.
  Light is the default; dark follows `prefers-color-scheme` unless the visitor
  picks a theme with the header toggle (stored in `localStorage`, key `scmle-theme`).
* `src/styles/site.css` (shell), `landing.css` (home) and `docs.css` (documentation).
* `src/layouts/BaseLayout.astro` owns metadata (canonical, Open Graph, Twitter,
  icons, manifest) and the header and footer; `DocsLayout.astro` adds the sidebar,
  filter and pager. `src/scripts/site.js` is the only client script (about 2 kB gzip).
* System font stack only: no third-party fonts are loaded.
* `public/og.png` and the touch icons are committed. Regenerate them with
  `node scripts/make-assets.mjs` (uses the Playwright Chromium already installed
  for the e2e tests).
* The sitemap and web manifest are generated at build time
  (`src/pages/sitemap.xml.ts`, `src/pages/site.webmanifest.ts`).
* Every number on the landing page comes from the repository README,
  `docs/BENCHMARKS.md`, or a real local run that is labelled as such. Keep it that way.

`npm run preview` backgrounds itself when it detects an AI-agent environment
(`AI_AGENT`, `CLAUDECODE`), which makes Playwright's web server exit early; unset
those variables when running `npm run test:e2e` locally.
