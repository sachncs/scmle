import { expect, test } from "@playwright/test";

const routes = [
  "./",
  "./docs/",
  "./docs/quickstart/",
  "./docs/data/",
  "./docs/modalities/",
  "./docs/capabilities/",
  "./docs/modeling/",
  "./docs/evidence/",
  "./docs/workflows/",
  "./docs/api/",
  "./docs/architecture/",
  "./docs/cards/",
  "./docs/operations/",
  "./docs/security/",
  "./docs/recovery/",
  "./docs/migrations/",
  "./docs/troubleshooting/",
  "./docs/extensions/",
  "./docs/methods/",
  "./docs/dependencies/",
  "./docs/rehearsal/",
  "./docs/observability/",
  "./docs/trail/",
  "./docs/open-items/",
  "./docs/leakage/",
  "./docs/breadth/",
  "./docs/start-here/",
  "./docs/glossary/",
  "./docs/decisions/",
  "./docs/reading-results/",
  "./docs/when-things-go-wrong/",
  "./docs/what-scmle-cannot-do/",
  "./docs/governed-data/",
  "./docs/qualification/",
  "./docs/causal-decision/",
  "./docs/api-stability/",
  "./docs/cli/",
  "./docs/artifacts/",
  "./docs/benchmarks/",
  "./docs/operating-loop/",
  "./docs/service/",
  "./docs/local/",
  "./docs/orchestration/",
  "./docs/portable-training/",
  "./docs/releasing/",
  "./docs/threat-model/",
  "./docs/tenant-isolation/",
];

test("every public route renders a standalone page", async ({ page }) => {
  test.setTimeout(120_000);
  for (const route of routes) {
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    await expect(page.locator("h1").first(), route).toBeVisible();
    await expect(page.locator('a[href$=".md"]'), route).toHaveCount(0);
  }
});

test("every page links out only to the repository home, issues and licence", async ({ page }) => {
  test.setTimeout(120_000);
  const allowed = new Set([
    "https://github.com/sachncs/scmle",
    "https://github.com/sachncs/scmle/issues",
    "https://www.apache.org/licenses/LICENSE-2.0",
    "https://pypi.org/project/scmle/",
  ]);
  for (const route of routes) {
    await page.goto(route);
    const hrefs = await page.$$eval("a[href]", (anchors) => anchors.map((a) => a.href));
    for (const href of hrefs) {
      const url = new URL(href);
      if (url.hostname === "127.0.0.1") continue;
      expect(allowed.has(`${url.origin}${url.pathname.replace(/\/$/, "")}`), `${route} -> ${href}`).toBe(true);
    }
  }
});

test("sidebar groups every page and the filter hides empty groups", async ({ page }) => {
  await page.goto("./docs/cli/");
  await expect(page.locator(".nav-group-label").first()).toBeVisible();
  await expect(page.locator('.sidebar nav a[aria-current="page"]')).toHaveText(/Command line/);
  await page.getByLabel("Filter documentation pages").fill("tenant");
  await expect(page.locator(".sidebar nav a:visible")).toHaveCount(1);
  await expect(page.locator(".nav-group:visible")).toHaveCount(1);
});

test("the glossary search hides non-matching terms and empty groups", async ({ page }) => {
  await page.goto("./docs/glossary/");
  const terms = page.locator(".term");
  const total = await terms.count();
  expect(total).toBeGreaterThan(40);
  await page.getByLabel("Search the glossary").fill("leak");
  const visible = await page.locator(".term:visible").count();
  expect(visible).toBeGreaterThan(0);
  expect(visible).toBeLessThan(total);
  await page.getByLabel("Search the glossary").fill("zzzzqqq");
  await expect(page.locator("#term-empty")).toBeVisible();
});

test("start here is linked from the home hero and leads the sidebar", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("main").getByRole("link", { name: /Start here/ }).first()).toHaveAttribute("href", "/scmle/docs/start-here/");
  await page.goto("./docs/start-here/");
  await expect(page.locator(".nav-group-label").first()).toHaveText("Learn");
  await expect(page.locator(".sidebar nav a").first()).toHaveText(/Start here/);
});

test("modeling page exposes the estimation contract and honest boundaries", async ({
  page,
}) => {
  await page.goto("./docs/modeling/");
  await expect(
    page.getByRole("heading", {
      name: "Parameter estimation is not hyperparameter tuning",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Parameter estimation contract" }),
  ).toBeVisible();
  await expect(page.getByText("Identifiability", { exact: true })).toBeVisible();
  await expect(page.getByText("reference calculation", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("There is no image, document, audio, video, graph, or geospatial JEPA", { exact: false })).toBeVisible();
  await expect(page.getByText("MLProject is not a production non-tabular modelling path", { exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "Use the public API" })).toHaveAttribute(
    "href",
    "/scmle/docs/api/",
  );
  await expect(page.locator('a[href*=".md"]')).toHaveCount(0);
});

test("documentation navigation exposes a keyboard focus path", async ({
  page,
}) => {
  await page.goto("./docs/architecture/");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveAttribute("href", "#main");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveAttribute(
    "aria-label",
    "SCMLE home",
  );
  // The sidebar links must be reachable by keyboard.
  let reached = false;
  for (let press = 0; press < 20 && !reached; press += 1) {
    await page.keyboard.press("Tab");
    reached = await page.evaluate(() => {
      const el = document.activeElement;
      return Boolean(el && el.closest(".sidebar nav") && /docs\//.test(el.getAttribute("href") ?? ""));
    });
  }
  expect(reached).toBe(true);
});

test("landing page tells the product story with base-aware calls to action", async ({
  page,
}) => {
  await page.goto("./");
  await expect(
    page.getByRole("heading", { level: 1, name: "Mathematics meets engineering." }),
  ).toBeVisible();
  const main = page.locator("main");
  await expect(main.getByRole("link", { name: "Get started" })).toHaveAttribute(
    "href",
    "/scmle/docs/quickstart/",
  );
  await expect(main.getByRole("link", { name: "View on GitHub" })).toHaveAttribute(
    "href",
    "https://github.com/sachncs/scmle",
  );
  for (const id of ["lifecycle", "capabilities", "quickstart", "benchmarks", "trust", "install"]) {
    await expect(page.locator(`#${id}`), id).toHaveCount(1);
  }
  await expect(page.getByText("In development", { exact: true })).toBeVisible();
  await expect(page.locator("main svg[role='img'] title").first()).toBeAttached();
});

test("landing page loads no third-party fonts or scripts", async ({ page }) => {
  const external = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.hostname !== "127.0.0.1") external.push(request.url());
  });
  await page.goto("./");
  await page.waitForLoadState("networkidle");
  expect(external).toEqual([]);
});

test("theme toggle switches, persists and respects the system scheme", async ({
  browser,
}) => {
  const context = await browser.newContext({ colorScheme: "light" });
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:4322/scmle/");
  const toggle = page.getByRole("button", { name: /Switch to dark theme/ });
  await expect(toggle).toBeVisible();
  await toggle.click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: /Switch to light theme/ })).toBeVisible();
  await context.close();

  const dark = await browser.newContext({ colorScheme: "dark" });
  const darkPage = await dark.newPage();
  await darkPage.goto("http://127.0.0.1:4322/scmle/");
  const background = await darkPage.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(background).toBe("rgb(11, 11, 13)");
  await dark.close();
});

test("reduced motion shows all content without reveal transitions", async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto("http://127.0.0.1:4322/scmle/");
  await expect(page.locator("#trust .reveal").first()).toHaveCSS("opacity", "1");
  await context.close();
});

test("docs pages offer anchors, copy buttons, filter and pager", async ({ page }) => {
  await page.goto("./docs/quickstart/");
  await expect(page.locator("h2 .anchor").first()).toHaveAttribute("href", /^#/);
  await expect(page.getByRole("button", { name: "Copy code to clipboard" }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: /Next/ })).toHaveAttribute("href", "/scmle/docs/data/");
  await page.getByLabel("Filter documentation pages").fill("secur");
  await expect(page.locator(".sidebar nav a:visible")).toHaveCount(1);
});

test("skip link is the first tab stop and focus is visible", async ({ page }) => {
  await page.goto("./");
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  await expect(skip).toBeVisible();
});

test("pages work at phone width without horizontal overflow", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of routes) {
    await page.goto(route);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    expect(overflow, route).toBeLessThanOrEqual(0);
  }
});

test("Escape closes the mobile menu and the docs chip row shows a scroll cue", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("./docs/cli/");
  const menu = page.locator("details.menu");
  await menu.locator("summary").click();
  await expect(menu).toHaveAttribute("open", "");
  await page.keyboard.press("Escape");
  await expect(menu).not.toHaveAttribute("open", "");
  const nav = page.locator(".sidebar nav");
  const overflows = await nav.evaluate((el) => el.scrollWidth > el.clientWidth);
  expect(overflows).toBe(true);
  await expect(nav).toHaveClass(/fade-(start|end)/);
});
