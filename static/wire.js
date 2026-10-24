// A pixelated wireframe hall behind the page: a tunnel of frames receding to a vanishing point,
// drawn at 1/3 resolution and scaled up with nearest-neighbour, so every line is chunky pixels.
// It drifts slowly forward; with reduced motion it is drawn once.
(function () {
  const c = document.createElement("canvas"); c.id = "wire"; c.setAttribute("aria-hidden", "true");
  Object.assign(c.style, {position: "fixed", inset: "0", width: "100vw", height: "100vh", zIndex: "-1", imageRendering: "pixelated", pointerEvents: "none"});
  const S = 3, x = c.getContext("2d");
  let W = 0, H = 0, t = 0, last = 0;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  function size() { W = Math.ceil(innerWidth / S); H = Math.ceil(innerHeight / S); c.width = W; c.height = H; x.imageSmoothingEnabled = false; draw(); }
  // a line drawn pixel by pixel (Bresenham), so it stays crisp when scaled up
  function px(x0, y0, x1, y1) {
    x0 |= 0; y0 |= 0; x1 |= 0; y1 |= 0;
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1; let e = dx + dy, n = 0;
    while (n++ < 4000) { if (x0 >= 0 && y0 >= 0 && x0 < W && y0 < H) x.fillRect(x0, y0, 1, 1); if (x0 === x1 && y0 === y1) break; const e2 = 2 * e; if (e2 >= dy) { e += dy; x0 += sx; } if (e2 <= dx) { e += dx; y0 += sy; } }
  }
  function draw() {
    const css = getComputedStyle(document.documentElement);
    x.fillStyle = css.getPropertyValue("--sky").trim() || "#d8ecfc"; x.fillRect(0, 0, W, H);
    x.fillStyle = css.getPropertyValue("--wire").trim() || "#4aa3f0";
    const vx = W * .64, vy = H * .40;
    // rays from the vanishing point
    for (let k = 0; k < 28; k++) { const a = k / 28 * Math.PI * 2 + .11; px(vx, vy, vx + Math.cos(a) * W * 1.4, vy + Math.sin(a) * H * 1.4); }
    // receding frames (the hall), moving towards the viewer
    const N = 9, phase = (t % 1);
    for (let i = 0; i < N; i++) {
      const z = (i + phase) / N, s = Math.pow(z, 2.2) * 1.6;           // 0 = far, 1 = near
      const w = W * s, h = H * s * .9, x0 = vx - w * .62, y0 = vy - h * .42;
      px(x0, y0, x0 + w, y0); px(x0 + w, y0, x0 + w, y0 + h); px(x0 + w, y0 + h, x0, y0 + h); px(x0, y0 + h, x0, y0);
    }
    // two slanted panels, like walls of the line
    for (let k = 0; k < 7; k++) {
      const f = k / 7; px(W * (.02 + .1 * f), H * .14, W * (.22 + .1 * f), H * .84); px(W * (.9 - .05 * f), H * .05, W * (.98 - .02 * f), H * .7);
    }
  }
  function loop(ts) { if (!document.hidden && ts - last > 60) { t += .012; last = ts; draw(); } requestAnimationFrame(loop); }
  function init() { document.body.prepend(c); size(); addEventListener("resize", size); if (!still) requestAnimationFrame(loop); }
  if (document.body) init(); else document.addEventListener("DOMContentLoaded", init);
})();
