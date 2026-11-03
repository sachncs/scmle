import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";

const siteRoot = resolve("dist");
const siteBase = "/scmle";
const siteOrigin = "https://sachncs.github.io";

function htmlFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) return htmlFiles(entryPath);
    return entry.name.endsWith(".html") ? [entryPath] : [];
  });
}

function pageUrl(filePath) {
  const pagePath = relative(siteRoot, filePath).replaceAll("\\", "/");
  if (pagePath === "index.html") return `${siteOrigin}${siteBase}/`;
  return `${siteOrigin}${siteBase}/${pagePath.replace(/\/index\.html$/, "")}`;
}

function targetFile(url) {
  const pathname = decodeURIComponent(url.pathname);
  if (!pathname.startsWith(`${siteBase}/`) && pathname !== siteBase) {
    return null;
  }
  const routePath = pathname.slice(siteBase.length).replace(/^\//, "");
  if (!routePath) return join(siteRoot, "index.html");
  const directPath = join(siteRoot, routePath);
  if (statSync(directPath, { throwIfNoEntry: false })?.isFile()) {
    return directPath;
  }
  const routeFile = join(siteRoot, routePath, "index.html");
  return statSync(routeFile, { throwIfNoEntry: false })?.isFile() ? routeFile : null;
}

function hasAnchor(contents, anchor) {
  const escapedAnchor = anchor.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`(?:id|name)=["']${escapedAnchor}["']`).test(contents);
}

const failures = [];
let checkedLinks = 0;
const files = htmlFiles(siteRoot);

for (const filePath of files) {
  const contents = readFileSync(filePath, "utf8");
  const sourceUrl = pageUrl(filePath);
  const links = contents.matchAll(/(?:href|src)=["']([^"']+)["']|<meta[^>]+(?:og:image|twitter:image)"\s+content="([^"]+)"/g);
  for (const [, hrefAttr, metaUrl] of links) {
    const href = hrefAttr ?? metaUrl;
    if (href === "#" || /^(mailto|tel|javascript):/i.test(href)) continue;
    const url = new URL(href, sourceUrl);
    if (url.origin !== siteOrigin) continue;
    checkedLinks += 1;
    const destination = targetFile(url);
    if (!destination) {
      failures.push(`${relative(siteRoot, filePath)} -> ${href}`);
      continue;
    }
    if (url.hash && !hasAnchor(readFileSync(destination, "utf8"), url.hash.slice(1))) {
      failures.push(`${relative(siteRoot, filePath)} -> ${href} (missing anchor)`);
    }
  }
}

if (failures.length) {
  console.error(`Found ${failures.length} broken internal links:`);
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(`Internal link checks passed for ${files.length} HTML files (${checkedLinks} links).`);
}
