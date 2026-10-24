// 師匠の一分講座, the master's one-minute lesson.
// A generated, narrated lesson that plays after a junior's mistake. Each lesson is a list of scenes; a scene draws
// itself into the stage, narrates one line (browser speech, in the language currently chosen) with subtitles, and
// can wait for an answer. Nothing is pre-recorded: the scenes are built from the master's captured rule and the part
// the junior got wrong.
//   Lesson.play({title_ja, title_en, scenes: [{ja, en, draw(stage), hold?, ask?}]}) -> Promise (resolves on close)
(function () {
  const css = document.createElement("style");
  css.textContent = `
  .ls-veil{position:fixed;inset:0;z-index:50;background:rgba(10,12,16,.82);display:grid;place-items:center;padding:16px}
  .ls{width:min(980px,100%);background:#151515;color:#fff;display:flex;flex-direction:column;box-shadow:0 0 0 3px #fff;
      clip-path:polygon(0 6px,6px 6px,6px 0,calc(100% - 6px) 0,calc(100% - 6px) 6px,100% 6px,100% calc(100% - 6px),calc(100% - 6px) calc(100% - 6px),calc(100% - 6px) 100%,6px 100%,6px calc(100% - 6px),0 calc(100% - 6px))}
  .ls-top{display:flex;align-items:center;gap:12px;padding:12px 16px;border-bottom:2px solid #333}
  .ls-top .rec{width:12px;height:12px;background:#e5432d;animation:lsblink 1s steps(2) infinite}
  .ls-top b{font-family:var(--px,monospace);font-weight:400;font-size:1.15rem;letter-spacing:.06em}
  .ls-top span.sub{opacity:.75;font-size:.85rem}
  .ls-top .gen{margin-left:auto;font-size:.75rem;background:#4aa3f0;color:#fff;padding:3px 8px}
  .ls-stage{position:relative;aspect-ratio:16/8.2;background:#0e0e10;overflow:hidden}
  .ls-cap{min-height:4.2em;padding:12px 18px 6px;font-size:1.12rem;line-height:1.5;text-align:center}
  .ls-cap .ls-2{display:block;font-size:.9rem;opacity:.75}
  .ls-bar{height:5px;background:#333;margin:0 16px}
  .ls-bar i{display:block;height:100%;width:0;background:#4aa3f0;transition:width .3s}
  .ls-ctl{display:flex;gap:8px;padding:10px 16px 14px;align-items:center}
  .ls-ctl button{border:0;background:#fff;color:#151515;font:600 .9rem/1 "Zen Kaku Gothic New",system-ui,sans-serif;padding:9px 14px;cursor:pointer}
  .ls-ctl button.ghost{background:transparent;color:#fff;box-shadow:inset 0 0 0 2px #fff}
  .ls-ctl .dots{margin-left:auto;display:flex;gap:5px}.ls-ctl .dots i{width:10px;height:10px;background:#444}.ls-ctl .dots i.on{background:#4aa3f0}.ls-ctl .dots i.done{background:#fff}
  @keyframes lsblink{50%{opacity:0}}
  /* scene helpers */
  .ls-stage .center{position:absolute;inset:0;display:grid;place-items:center}
  .ls-big{font-family:var(--px,monospace);font-size:clamp(1.6rem,4vw,2.8rem);text-align:center;line-height:1.25}
  .ls-zoom{position:absolute;inset:6% 26%;outline:4px solid #fff;overflow:hidden;background:#000}
  .ls-zoom img{width:100%;height:100%;object-fit:cover;image-rendering:pixelated;animation:lszoom 4s ease-out forwards}
  @keyframes lszoom{from{transform:scale(1)}to{transform:scale(1.35)}}
  .ls-tag{position:absolute;padding:6px 12px;font-family:var(--px,monospace);font-size:1.3rem;background:#fff;color:#151515}
  .ls-tag.ng{background:#e5432d;color:#fff}.ls-tag.ok{background:#1f9d57;color:#fff}
  .ls-strip{position:absolute;left:4%;right:4%;top:22%;display:grid;grid-template-columns:repeat(8,1fr);gap:6px}
  .ls-strip div{position:relative;aspect-ratio:1;outline:3px solid #555;background:#000;transition:outline-color .25s}
  .ls-strip img{width:100%;height:100%;object-fit:cover;image-rendering:pixelated;display:block}
  .ls-strip div.ok{outline-color:#1f9d57}.ls-strip div.ng{outline-color:#e5432d}
  .ls-strip div b{position:absolute;right:4px;top:2px;font-size:1.2rem}.ls-strip div.ok b{color:#7fe0a6}.ls-strip div.ng b{color:#ff9a90}
  .ls-strip div span{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.6);font-size:.72rem;text-align:center}
  .ls-line{position:absolute;top:14%;bottom:18%;width:4px;margin-left:-2px;transition:left 2.4s cubic-bezier(.5,0,.2,1)}
  .ls-line.man{background:repeating-linear-gradient(#aaa 0 6px,transparent 6px 12px)}.ls-line.mas{background:#fff;box-shadow:0 0 12px #4aa3f0}
  .ls-line em{position:absolute;bottom:-26px;left:50%;transform:translateX(-50%);white-space:nowrap;font-style:normal;font-size:.85rem;background:#151515;padding:1px 6px}
  .ls-pair{position:absolute;inset:8% 10% 12%;display:grid;grid-template-columns:1fr 1fr;gap:6%}
  .ls-pair div{position:relative;outline:4px solid;background:#000}.ls-pair div.ok{outline-color:#1f9d57}.ls-pair div.ng{outline-color:#e5432d}
  .ls-pair img{width:100%;height:100%;object-fit:cover;image-rendering:pixelated;display:block}
  .ls-pair .ls-tag{left:8px;top:8px}
  .ls-ask{position:absolute;left:0;right:0;bottom:6%;display:flex;gap:14px;justify-content:center}
  .ls-ask button{border:0;font:400 1.4rem/1 var(--px,monospace);padding:12px 26px;cursor:pointer}
  .ls-ask .x{background:#e5432d;color:#fff}.ls-ask .o{background:#1f9d57;color:#fff}
  .ls-wave{position:absolute;left:8%;right:8%;height:30%;display:flex;align-items:center;gap:3px}
  .ls-wave i{flex:1;background:#4aa3f0;min-height:3px;transition:height .08s}
  .ls-wave.bad i{background:#e5432d}
  .ls-lab{position:absolute;left:8%;font-family:var(--px,monospace);font-size:1.2rem}
  @media (prefers-reduced-motion:reduce){.ls-zoom img{animation:none}.ls-line{transition:none}}`;
  document.head.appendChild(css);

  const lang = () => (document.body.classList.contains("lang-en") ? "en" : "ja");
  let muted = false, voices = [];
  const loadVoices = () => { try { voices = speechSynthesis.getVoices(); } catch (e) {} };
  try { loadVoices(); speechSynthesis.onvoiceschanged = loadVoices; } catch (e) {}

  function say(text, l) {
    return new Promise(res => {
      const est = Math.max(1800, text.length * (l === "ja" ? 140 : 65));
      if (muted || !("speechSynthesis" in window)) return setTimeout(res, est);
      try {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text); u.lang = l === "ja" ? "ja-JP" : "en-US"; u.rate = l === "ja" ? 1.05 : 1.0;
        const v = voices.find(v => v.lang && v.lang.toLowerCase().startsWith(l === "ja" ? "ja" : "en")); if (v) u.voice = v;
        let done = false; const fin = () => { if (!done) { done = true; res(); } };
        u.onend = fin; u.onerror = fin; setTimeout(fin, est + 4000);
        speechSynthesis.speak(u);
      } catch (e) { setTimeout(res, est); }
    });
  }

  function play(lesson) {
    return new Promise(resolve => {
      const veil = document.createElement("div"); veil.className = "ls-veil";
      veil.innerHTML = `<div class="ls" role="dialog" aria-modal="true" aria-label="Lesson">
        <div class="ls-top"><i class="rec" aria-hidden="true"></i><b>${lesson.title_ja}</b><span class="sub">${lesson.title_en}</span><span class="gen">自動生成 · generated just now</span></div>
        <div class="ls-stage"></div><div class="ls-cap" aria-live="polite"></div><div class="ls-bar"><i></i></div>
        <div class="ls-ctl"><button class="ghost" data-a="replay">↺ もう一度 · Replay</button><button class="ghost" data-a="mute">🔈 音声 · Voice</button><button data-a="close">閉じる · Done</button><div class="dots"></div></div></div>`;
      document.body.appendChild(veil);
      const stage = veil.querySelector(".ls-stage"), cap = veil.querySelector(".ls-cap"), bar = veil.querySelector(".ls-bar i"), dots = veil.querySelector(".dots");
      const S = lesson.scenes; dots.innerHTML = S.map(() => "<i></i>").join("");
      let run = 0, closed = false;
      const close = () => { if (closed) return; closed = true; try { speechSynthesis.cancel(); } catch (e) {} veil.remove(); resolve(); };
      veil.querySelector('[data-a="close"]').onclick = close;
      veil.querySelector('[data-a="mute"]').onclick = e => { muted = !muted; e.currentTarget.textContent = muted ? "🔇 音声オフ · Voice off" : "🔈 音声 · Voice"; if (muted) try { speechSynthesis.cancel(); } catch (er) {} };
      veil.querySelector('[data-a="replay"]').onclick = () => go();
      document.addEventListener("keydown", function k(e) { if (closed) return document.removeEventListener("keydown", k); if (e.key === "Escape") close(); });
      async function go() {
        const me = ++run;
        for (let i = 0; i < S.length; i++) {
          if (me !== run || closed) return;
          [...dots.children].forEach((d, k) => d.className = k < i ? "done" : k === i ? "on" : "");
          bar.style.width = (100 * i / S.length) + "%";
          const sc = S[i], l = lang();
          stage.innerHTML = ""; const answered = sc.draw ? sc.draw(stage) : null;
          cap.innerHTML = l === "ja" ? `<span class="ls-1">${sc.ja}</span><span class="ls-2">${sc.en}</span>` : `<span class="ls-1">${sc.en}</span><span class="ls-2">${sc.ja}</span>`;
          await Promise.all([say(l === "ja" ? sc.ja : sc.en, l), new Promise(r => setTimeout(r, sc.hold || 1200))]);
          if (sc.after) { await sc.after(stage); if (me !== run || closed) return; }
          if (answered && answered.then) {               // a check question: wait for the answer, then say the verdict
            const v = await answered; if (me !== run || closed) return;
            cap.innerHTML = l === "ja" ? `<span class="ls-1">${v.ja}</span><span class="ls-2">${v.en}</span>` : `<span class="ls-1">${v.en}</span><span class="ls-2">${v.ja}</span>`;
            await say(l === "ja" ? v.ja : v.en, l); await new Promise(r => setTimeout(r, 900));
          }
        }
        if (me === run && !closed) { bar.style.width = "100%"; [...dots.children].forEach(d => d.className = "done"); }
      }
      go();
    });
  }

  // a fake-but-honest waveform: bars driven by the real audio element's analyser when available
  function wave(stage, audio, bad, top) {
    const w = document.createElement("div"); w.className = "ls-wave" + (bad ? " bad" : ""); w.style.top = top; stage.appendChild(w);
    const N = 48; w.innerHTML = "<i></i>".repeat(N); const bars = [...w.children];
    let an = null, data = null;
    try { const AC = window.__lsAC || (window.__lsAC = new (window.AudioContext || window.webkitAudioContext)()); const src = AC.createMediaElementSource(audio);
      an = AC.createAnalyser(); an.fftSize = 128; src.connect(an); an.connect(AC.destination); data = new Uint8Array(an.frequencyBinCount); AC.resume && AC.resume(); } catch (e) {}
    (function tick() { if (!w.isConnected) return;
      if (an) { an.getByteFrequencyData(data); bars.forEach((b, i) => b.style.height = (4 + data[Math.floor(i * data.length / N)] / 255 * 100) + "%"); }
      else bars.forEach(b => b.style.height = (4 + Math.random() * (audio.paused ? 4 : 60)) + "%");
      requestAnimationFrame(tick); })();
    return w;
  }

  window.Lesson = {play, wave};
})();
