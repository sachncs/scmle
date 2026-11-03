// Small progressive enhancements. The site is fully usable without this file.
const root = document.documentElement;

// Theme toggle -------------------------------------------------------------
const toggle = document.querySelector("[data-theme-toggle]");
const dark = window.matchMedia("(prefers-color-scheme: dark)");
const effective = () => root.dataset.theme || (dark.matches ? "dark" : "light");
function syncToggle() {
  if (!toggle) return;
  const next = effective() === "dark" ? "light" : "dark";
  toggle.setAttribute("aria-label", `Switch to ${next} theme`);
  toggle.setAttribute("title", `Switch to ${next} theme`);
}
toggle?.addEventListener("click", () => {
  const next = effective() === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  try { localStorage.setItem("scmle-theme", next); } catch { /* storage unavailable */ }
  syncToggle();
});
dark.addEventListener?.("change", syncToggle);
syncToggle();

// Close the mobile menu after choosing a link ------------------------------
document.querySelectorAll(".menu-panel a").forEach((a) =>
  a.addEventListener("click", () => a.closest("details")?.removeAttribute("open")),
);

// Scroll reveal ------------------------------------------------------------
const reveals = document.querySelectorAll(".reveal");
if (reveals.length) {
  if ("IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }),
      { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
    );
    reveals.forEach((el) => io.observe(el));
  } else {
    reveals.forEach((el) => el.classList.add("in"));
  }
}

// Documentation helpers ----------------------------------------------------
const docs = document.querySelector(".docs-content");
if (docs) {
  const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  const used = new Set([...docs.querySelectorAll("[id]")].map((n) => n.id));
  docs.querySelectorAll("h2, h3").forEach((h) => {
    if (!h.id) {
      let id = slug(h.textContent || "");
      if (!id) return;
      for (let n = 2; used.has(id); n += 1) id = `${slug(h.textContent || "")}-${n}`;
      h.id = id;
    }
    used.add(h.id);
    const a = document.createElement("a");
    a.className = "anchor";
    a.href = `#${h.id}`;
    a.textContent = "#";
    a.setAttribute("aria-label", "Link to this section");
    a.setAttribute("aria-describedby", h.id);
    h.append(a);
  });

  const live = document.createElement("div");
  live.className = "sr-only";
  live.setAttribute("aria-live", "polite");
  document.body.append(live);
  docs.querySelectorAll("pre").forEach((pre) => {
    pre.tabIndex = 0;
    const wrap = document.createElement("div");
    wrap.className = "code-wrap";
    pre.replaceWith(wrap);
    wrap.append(pre);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy";
    btn.textContent = "Copy";
    btn.setAttribute("aria-label", "Copy code to clipboard");
    btn.addEventListener("click", async () => {
      const text = pre.innerText;
      try { await navigator.clipboard.writeText(text); }
      catch {
        const ta = document.createElement("textarea");
        ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove();
      }
      btn.textContent = "Copied";
      live.textContent = "Code copied to clipboard";
      setTimeout(() => { btn.textContent = "Copy"; live.textContent = ""; }, 1800);
    });
    wrap.append(btn);
  });
  docs.querySelectorAll(".capability-table, .model-table").forEach((t) => {
    if (!t.hasAttribute("tabindex")) t.tabIndex = 0;
    if (!t.hasAttribute("role")) t.setAttribute("role", "region");
    if (!t.hasAttribute("aria-label")) t.setAttribute("aria-label", "Scrollable table");
  });
}

// Sidebar: filter + keep the active chip in view ---------------------------
const input = document.querySelector("#doc-filter");
const links = [...document.querySelectorAll(".sidebar nav a")];
const sideNav = document.querySelector(".sidebar nav");
const active = document.querySelector('.sidebar nav a[aria-current="page"]');
if (sideNav && active && sideNav.scrollWidth > sideNav.clientWidth && getComputedStyle(sideNav).overflowX !== "visible") {
  sideNav.scrollLeft = active.offsetLeft - (sideNav.clientWidth - active.offsetWidth) / 2;
}
if (input) {
  const empty = document.querySelector(".filter-empty");
  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    let shown = 0;
    links.forEach((a) => {
      const hit = !q || a.textContent.toLowerCase().includes(q);
      a.hidden = !hit;
      if (hit) shown += 1;
    });
    if (empty) empty.hidden = shown > 0;
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") links.find((a) => !a.hidden)?.click();
  });
  addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
    if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey) { e.preventDefault(); input.focus(); }
  });
}
