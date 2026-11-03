import { readdir, readFile } from "node:fs/promises";

const root = new URL("../dist/", import.meta.url);

async function htmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directory);
    if (entry.isDirectory()) {
      files.push(...(await htmlFiles(path)));
    } else if (entry.name.endsWith(".html")) {
      files.push(path);
    }
  }
  return files;
}

function countMatches(source, expression) {
  return [...source.matchAll(expression)].length;
}

function visibleText(source) {
  return source
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-z]+;/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const files = await htmlFiles(root);
const home = await readFile(new URL("index.html", root), "utf8");
if (!/In development/.test(home)) {
  console.error("index.html: the qualification instrument must be labelled 'In development'");
  process.exit(1);
}
if (/\bsealed\b|guaranteed|perfect/i.test(visibleText(home.replace(/<script[\s\S]*?<\/script>|<style[\s\S]*?<\/style>/g, "")))) {
  console.error("index.html: contains a claim word that the honesty rules forbid (sealed, guaranteed, perfect)");
  process.exit(1);
}
const failures = [];
for (const file of files) {
  const source = await readFile(file, "utf8");
  const label = file.pathname.replace(root.pathname, "");
  if (!/<html\b[^>]*\blang=["'][^"']+["']/i.test(source)) {
    failures.push(`${label}: missing html lang`);
  }
  if (!/<title>[^<]+<\/title>/i.test(source)) {
    failures.push(`${label}: missing non-empty title`);
  }
  if (countMatches(source, /<h1\b/gi) !== 1) {
    failures.push(`${label}: expected exactly one h1`);
  }
  if (!/<meta\s+name="viewport"[^>]*width=device-width/i.test(source)) {
    failures.push(`${label}: missing responsive viewport meta`);
  }
  if (!/<meta\s+name="description"\s+content="[^"]{20,}"/i.test(source)) {
    failures.push(`${label}: missing meta description`);
  }
  if (!/<link\s+rel="canonical"\s+href="https:\/\/sachncs\.github\.io\/scmle\//i.test(source)) {
    failures.push(`${label}: missing base-aware canonical link`);
  }
  for (const property of ["og:title", "og:description", "og:image", "twitter:card"]) {
    if (!new RegExp(`<meta\\s+(?:property|name)="${property}"`, "i").test(source)) {
      failures.push(`${label}: missing ${property} meta`);
    }
  }
  if (countMatches(source, /<main\b/gi) !== 1) {
    failures.push(`${label}: expected exactly one main landmark`);
  }
  if (!/class="skip"\s+href="#main"/.test(source)) {
    failures.push(`${label}: missing skip link to #main`);
  }
  if (/fonts\.googleapis\.com|fonts\.gstatic\.com/.test(source)) {
    failures.push(`${label}: loads third-party fonts`);
  }
  for (const labelled of source.matchAll(/aria-labelledby="([^"]+)"/g)) {
    for (const id of labelled[1].split(/\s+/)) {
      if (!new RegExp(`\\bid="${id}"`).test(source)) {
        failures.push(`${label}: aria-labelledby points to missing id ${id}`);
      }
    }
  }
  for (const svg of source.matchAll(/<svg\b([^>]*)>/gi)) {
    if (/role="img"/.test(svg[1]) && !/aria-label(?:ledby)?=/.test(svg[1])) {
      failures.push(`${label}: svg role=img without an accessible name`);
    }
  }
  for (const image of source.matchAll(/<img\b[^>]*>/gi)) {
    if (!/\balt=["'][^"']*["']/i.test(image[0])) {
      failures.push(`${label}: image missing alt text`);
    }
  }
  for (const link of source.matchAll(/<a\b([^>]*)>([\s\S]*?)<\/a>/gi)) {
    const attributes = link[1];
    const text = visibleText(link[2]);
    const labelled = /\baria-label=["'][^"']+["']/i.test(attributes);
    if (!text && !labelled) {
      failures.push(`${label}: link has no accessible name`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log(`Accessibility smoke checks passed for ${files.length} HTML files.`);
