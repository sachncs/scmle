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
