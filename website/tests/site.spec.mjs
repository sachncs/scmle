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
  "./docs/release-evidence/",
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

test("release evidence page keeps the beta boundary and generated checks visible", async ({
  page,
}) => {
  await page.goto("./docs/release-evidence/");
  await expect(page.getByRole("heading", { name: /Make the beta/ })).toBeVisible();
  await expect(page.getByRole("region", { name: "Capability evidence view" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Tabular beta evidence board" })).toBeVisible();
  await expect(page.getByText("Independent review", { exact: true })).toBeVisible();
  await expect(page.getByText("experimental", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("non-tabular lifecycles remain experimental", { exact: false })).toBeVisible();
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
