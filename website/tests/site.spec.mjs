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
];

test("every public route renders a standalone page", async ({ page }) => {
  for (const route of routes) {
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    await expect(page.locator("h1").first(), route).toBeVisible();
    await expect(page.locator('a[href$=".md"]'), route).toHaveCount(0);
  }
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
  await expect(page.locator(":focus")).toHaveAttribute(
    "aria-label",
    "SCMLE home",
  );
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await expect(page.locator(":focus")).toHaveAttribute("href", /docs\//);
});
