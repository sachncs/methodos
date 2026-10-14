import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("product page exposes the primary journey", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveTitle("methodos — procedural memory for agents");
  await expect(page.getByRole("heading", { name: /Give your agent a way forward/ })).toBeVisible();
  await expect(page.getByRole("img", { name: "Procedural graph" })).toBeVisible();
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
    "href",
    "https://sachncs.github.io/methodos/",
  );
  await expect(page.locator('meta[property="og:title"]')).toHaveAttribute("content", /methodos/);
  await expect(page.getByRole("link", { name: "Read the docs ↗" })).toHaveAttribute(
    "href",
    /github.com\/sachncs\/methodos/,
  );
  await page.getByRole("link", { name: "See how it works ↓" }).click();
  await expect(page.locator("#loop")).toBeInViewport();
  await expect(page.getByRole("heading", { name: "Useful in the hot path. Improving off the hot path." })).toBeVisible();
});

test("navigation and accessibility landmarks remain available on mobile", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("banner")).toBeVisible();
  await expect(page.getByRole("main")).toBeVisible();
  await expect(page.getByRole("contentinfo")).toBeVisible();
  await expect(page.getByRole("link", { name: "Skip to content" })).toHaveAttribute(
    "href",
    "#main-content",
  );
  await expect(page.locator('a[href="#architecture"]')).toHaveAttribute(
    "href",
    "#architecture",
  );
  await expect(page.locator('a[href="#start"]')).toHaveAttribute("href", "#start");
});

test("product page stays within the accessibility budget", async ({ page }) => {
  await page.goto("/");
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("branded not-found route and crawl metadata are available", async ({ page }) => {
  const notFound = await page.goto("/404.html");
  expect(notFound?.status()).toBe(200);
  await expect(page).toHaveTitle("Page not found — methodos");
  await expect(page.getByRole("heading", { name: "This path doesn’t exist." })).toBeVisible();
  await expect(page.getByRole("link", { name: /Back to methodos/ })).toHaveAttribute(
    "href",
    "/methodos/",
  );
  const robots = await page.request.get("/robots.txt");
  const sitemap = await page.request.get("/sitemap.xml");
  expect(robots.status()).toBe(200);
  expect(sitemap.status()).toBe(200);
});

test("product composition stays visually stable", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".visual")).toHaveScreenshot("product-visual.png", {
    animations: "disabled",
    caret: "hide",
    maxDiffPixelRatio: 0.05,
  });
});
