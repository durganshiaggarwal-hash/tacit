// Two languages, one switch. Every piece of text is written as Japanese followed by <span class="en">English</span>.
// This script wraps the Japanese part in <span class="ja">, then shows one language and offers the other on hover
// (or tap). The switch in the corner flips between 日本語 and English; the choice is remembered on this device.
(function () {
  const KEY = "tacit-lang";
  let lang = "ja";
  try { lang = localStorage.getItem(KEY) || "ja"; } catch (e) {}
  if (lang !== "en") lang = "ja";

  const css = document.createElement("style");
  css.textContent = `
    body.lang-ja .en{display:none!important}
    body.lang-en .ja{display:none!important}
    body.lang-en .en{font-size:inherit!important;color:inherit!important;opacity:1!important;letter-spacing:inherit!important;font-weight:inherit!important}
    .lang-tip{position:fixed;z-index:9999;max-width:min(440px,92vw);background:#10141c;color:#fff;font:500 14px/1.5 "Zen Kaku Gothic New","Hiragino Sans",system-ui,sans-serif;
      padding:8px 12px;pointer-events:none;opacity:0;transform:translateY(4px);transition:opacity .12s,transform .12s;text-align:left}
    .lang-tip.show{opacity:1;transform:none}
    .lang-tip::before{content:attr(data-l);font-size:10px;letter-spacing:.08em;opacity:.55;margin-right:8px}
    .lang-switch{position:fixed;right:14px;bottom:14px;z-index:9998;display:flex;background:#fff;box-shadow:inset 0 0 0 3px #151515,0 4px 14px rgba(0,0,0,.18);padding:3px}
    .lang-switch button{border:0;background:transparent;color:#151515;font:600 14px/1 "Zen Kaku Gothic New","Hiragino Sans",system-ui,sans-serif;padding:9px 14px;cursor:pointer}
    .lang-switch button[aria-pressed="true"]{background:#151515;color:#fff}
    @media (prefers-reduced-motion:reduce){.lang-tip{transition:none}}`;
  document.head.appendChild(css);

  const JA = /[\u3040-\u30ff\u3400-\u9fff]/;
  // wrap the Japanese text that sits next to an English span, so either can be hidden
  function wrap(root) {
    const scope = root.querySelectorAll ? [root, ...root.querySelectorAll("*")] : [];
    for (const el of scope) {
      if (!el.children || el.classList.contains("ja") || el.classList.contains("en")) continue;
      let hasEn = false; for (const c of el.children) if (c.classList.contains("en")) { hasEn = true; break; }
      if (!hasEn) continue;
      for (const n of [...el.childNodes]) {
        if (n.nodeType === 3 && n.textContent.trim()) { const s = document.createElement("span"); s.className = "ja"; n.replaceWith(s); s.appendChild(n); }
        // a sibling element holding Japanese words (a <b>, a tag) belongs to the Japanese side too
        else if (n.nodeType === 1 && !n.classList.contains("en") && !n.classList.contains("ja") && !n.matches("kbd,img,svg,input,canvas,.sym,.pts,.keep")
                 && JA.test(n.textContent) && !n.querySelector(".en")) n.classList.add("ja");
      }
    }
  }
  let queued = false;
  const mo = new MutationObserver(() => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; mo.disconnect(); wrap(document.body); watch(); }); });
  const watch = () => mo.observe(document.body, {childList: true, subtree: true, characterData: false});

  // hover: show the other language
  const tip = document.createElement("div"); tip.className = "lang-tip"; tip.setAttribute("role", "tooltip");
  let current = null, hideT = null;
  function holder(el) {
    const want = lang === "ja" ? "en" : "ja";
    while (el && el !== document.body) {
      if (el.children) for (const c of el.children) if (c.classList && c.classList.contains(want) && c.textContent.trim()) return [el, el.querySelectorAll(":scope > ." + want)];
      el = el.parentElement;
    }
    return null;
  }
  function show(h) {
    if (!h) return hide();
    const [el, parts] = h; if (current === el) return; current = el;
    tip.textContent = [...parts].map(p => p.textContent.trim()).join(" ");
    tip.dataset.l = lang === "ja" ? "EN" : "日本語";
    const r = el.getBoundingClientRect(); tip.style.left = "0px"; tip.style.top = "0px"; tip.classList.add("show");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - tw / 2), innerWidth - tw - 8);
    let y = r.bottom + 8; if (y + th > innerHeight - 8) y = Math.max(8, r.top - th - 8);
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hide() { current = null; tip.classList.remove("show"); }
  document.addEventListener("pointerover", e => { if (e.pointerType !== "touch" && !e.target.closest(".lang-switch")) show(holder(e.target)); });
  document.addEventListener("pointerout", e => { if (e.pointerType !== "touch" && current && !current.contains(e.relatedTarget)) hide(); });
  document.addEventListener("pointerdown", e => {
    if (e.pointerType !== "touch" || e.target.closest(".lang-switch")) return;
    current = null; show(holder(e.target)); clearTimeout(hideT); hideT = setTimeout(hide, 2600);
  });
  window.addEventListener("scroll", hide, {passive: true});

  // the switch
  const sw = document.createElement("div"); sw.className = "lang-switch"; sw.setAttribute("role", "group"); sw.setAttribute("aria-label", "Language");
  sw.innerHTML = `<button type="button" data-l="ja">日本語</button><button type="button" data-l="en">English</button>`;
  sw.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; lang = b.dataset.l; try { localStorage.setItem(KEY, lang); } catch (er) {} apply(); });
  function apply() {
    document.body.classList.toggle("lang-ja", lang === "ja"); document.body.classList.toggle("lang-en", lang === "en");
    document.documentElement.lang = lang;
    sw.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.l === lang));
    hide();
  }
  function init() { wrap(document.body); watch(); document.body.appendChild(tip); document.body.appendChild(sw); apply(); }
  if (document.body) init(); else document.addEventListener("DOMContentLoaded", init);
})();
