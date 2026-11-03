import type { APIRoute } from "astro";

const origin = "https://sachncs.github.io";

export const GET: APIRoute = () => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const pages = Object.keys(import.meta.glob("./**/*.astro"))
    .map((file) => file.replace(/^\.\//, "").replace(/\.astro$/, ""))
    .filter((route) => route !== "404")
    .map((route) => (route === "index" ? "/" : `/${route.replace(/\/index$/, "")}/`))
    .sort();
  const urls = pages.map((path) => `  <url><loc>${origin}${base}${path}</loc></url>`).join("\n");
  const body = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;
  return new Response(body, { headers: { "Content-Type": "application/xml" } });
};
