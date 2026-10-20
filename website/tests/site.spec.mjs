import { expect, test } from "@playwright/test";

const routes = [
  "./",
  "./docs/",
  "./docs/quickstart/",
  "./docs/data/",
  "./docs/capabilities/",
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
];

test("every public route renders a standalone page", async ({ page }) => {
  for (const route of routes) {
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    await expect(page.locator("h1").first(), route).toBeVisible();
    await expect(page.locator('a[href$=".md"]'), route).toHaveCount(0);
  }
});

test("documentation navigation exposes a keyboard focus path", async ({
  page,
}) => {
  await page.goto("./docs/architecture/");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveAttribute(
    "aria-label",
    "super-charge-ml home",
  );
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveAttribute("href", /docs\//);
});
