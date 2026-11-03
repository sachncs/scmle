import type { APIRoute } from "astro";

export const GET: APIRoute = () => {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  const manifest = {
    name: "SCMLE",
    short_name: "SCMLE",
    description: "A curation-first machine learning lifecycle with leakage-safe evaluation and signed artifacts.",
    start_url: `${base}/`,
    scope: `${base}/`,
    display: "browser",
    background_color: "#fbfbfd",
    theme_color: "#0b0b0d",
    icons: [
      { src: `${base}/icon-192.png`, sizes: "192x192", type: "image/png" },
      { src: `${base}/icon-512.png`, sizes: "512x512", type: "image/png" },
      { src: `${base}/favicon.svg`, sizes: "any", type: "image/svg+xml", purpose: "any" },
    ],
  };
  return new Response(JSON.stringify(manifest, null, 2), { headers: { "Content-Type": "application/manifest+json" } });
};
