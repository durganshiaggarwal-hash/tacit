// Fills the status line under each step of the flow from the live passport summary.
(function () {
  const set = (k, t) => { const e = document.querySelector(`.tsteps [data-step="${k}"] .st`); if (e) e.textContent = t; };
  fetch("/api/passport").then(r => r.ok ? r.json() : null).then(p => {
    if (!p) return;
    const v = Object.values(p.senses), ans = v.reduce((a, s) => a + s.master_answers, 0), appr = v.reduce((a, s) => a + s.approved, 0);
    set("capture", ans ? `${ans} answers from the master` : "Start here");
    set("rules", p.total ? `${appr} of ${p.total} approved` : "None yet");
    set("train", p.total ? `${p.held} of ${p.total} held by a junior` : "Needs rules first");
    set("prince", p.total ? `${p.total} margin notes` : "No notes yet");
    set("passport", `${p.certified.length} of 3 senses certified`);
  }).catch(() => {});
})();
