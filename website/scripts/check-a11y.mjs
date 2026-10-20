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
