import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Every route in the sitemap, scanned with axe-core in two states:
// desktop in the light scheme and a phone in the dark scheme. Any violation fails.
const states = [
  { name: "desktop-light", viewport: { width: 1440, height: 900 }, colorScheme: "light" },
  { name: "mobile-dark", viewport: { width: 390, height: 844 }, colorScheme: "dark" },
];

async function sitemapPaths(request) {
  const response = await request.get("sitemap.xml");
  expect(response.status()).toBe(200);
  const body = await response.text();
  const paths = [...body.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => new URL(m[1]).pathname.replace(/^\/scmle\//, ""));
  expect(paths.length).toBeGreaterThan(40);
  return paths;
}

for (const state of states) {
  test(`axe finds no violations on any sitemap route (${state.name})`, async ({ browser, request }) => {
    test.setTimeout(240_000);
    const paths = await sitemapPaths(request);
    const context = await browser.newContext({ viewport: state.viewport, colorScheme: state.colorScheme });
    const page = await context.newPage();
    const failures = [];
    for (const path of paths) {
      await page.goto(`http://127.0.0.1:4322/scmle/${path}`);
      await page.waitForLoadState("networkidle");
      const { violations } = await new AxeBuilder({ page }).analyze();
      for (const v of violations) {
        failures.push(`${path || "/"} [${v.impact}] ${v.id}: ${v.nodes.length} node(s), e.g. ${v.nodes[0].target.join(" ")}`);
      }
    }
    await context.close();
    expect(failures, failures.join("\n")).toEqual([]);
  });
}
