/** An intentionally procedural interface preview, never model output or persistent world state. */
export interface PreviewView {
  travel: number;
  yaw: number;
  pitch: number;
  hue: number;
  rain: boolean;
}

export function drawPreview(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  time: number,
  view: PreviewView,
) {
  const w = width,
    h = height;
  const horizon = h * (0.48 + view.pitch * 0.12);
  const hue = view.hue;
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, `hsl(${hue}, 34%, 7%)`);
  sky.addColorStop(0.47, `hsl(${hue + 13}, 24%, 27%)`);
  sky.addColorStop(0.7, `hsl(${hue - 13}, 24%, 9%)`);
  sky.addColorStop(1, "#030708");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);
  const sunX = w * (0.68 - view.yaw * 0.12);
  const glow = ctx.createRadialGradient(sunX, horizon * 0.72, 2, sunX, horizon * 0.72, h * 0.4);
  glow.addColorStop(0, "rgba(239,206,159,.19)");
  glow.addColorStop(1, "rgba(239,206,159,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#e4d8b6";
  ctx.beginPath();
  ctx.arc(sunX, horizon * 0.66, h * 0.036, 0, Math.PI * 2);
  ctx.fill();
  for (let layer = 0; layer < 5; layer++) {
    const base = horizon + layer * h * 0.049;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w + 8; x += 8) {
      const p =
        (x / w) * 6 + layer * 2.7 + view.yaw * (layer + 1) * 0.2 + view.travel * 0.0003 * layer;
      const peak =
        Math.sin(p * 1.12) * 0.44 + Math.sin(p * 2.7 + 1) * 0.22 + Math.sin(p * 5.2) * 0.1;
      ctx.lineTo(x, base - (peak + 0.6) * h * (0.24 - layer * 0.021));
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = `hsl(${hue - layer * 5}, ${15 + layer * 2}%, ${23 - layer * 3.8}%)`;
    ctx.fill();
  }
  // A stylized corridor makes movement bindings testable without implying model inference.
  const vanishX = w * (0.5 - Math.sin(view.yaw) * 0.12);
  const ground = h * 0.66;
  ctx.beginPath();
  ctx.moveTo(vanishX - 3, ground);
  ctx.lineTo(vanishX + 3, ground);
  ctx.bezierCurveTo(w * 0.54, h * 0.81, w * 0.69, h * 0.9, w * 0.75, h);
  ctx.lineTo(w * 0.2, h);
  ctx.bezierCurveTo(w * 0.34, h * 0.88, w * 0.51, h * 0.78, vanishX - 3, ground);
  const road = ctx.createLinearGradient(0, ground, 0, h);
  road.addColorStop(0, "#607771");
  road.addColorStop(1, "#172826");
  ctx.fillStyle = road;
  ctx.fill();
  for (let i = 0; i < 32; i++) {
    const depth = (((i / 32 + view.travel * 0.0005) % 1) + 1) % 1;
    const y = ground + depth * depth * (h - ground);
    const spread = depth * depth * w * 0.6;
    for (const side of [-1, 1]) {
      const x = vanishX + side * (spread + 8) + Math.sin(i * 13.7) * depth * w * 0.06;
      const treeH = 4 + depth * h * 0.32;
      ctx.fillStyle = "#071814";
      ctx.beginPath();
      ctx.moveTo(x, y - treeH);
      ctx.lineTo(x - treeH * 0.23, y);
      ctx.lineTo(x + treeH * 0.23, y);
      ctx.fill();
      ctx.fillStyle = "rgba(117,177,152,.27)";
      ctx.fillRect(x, y - treeH * 0.4, Math.max(1, depth * 2), treeH * 0.4);
    }
  }
  const mist = ctx.createLinearGradient(0, horizon, 0, h * 0.78);
  mist.addColorStop(0, "rgba(150,172,168,0)");
  mist.addColorStop(0.45, "rgba(150,172,168,.12)");
  mist.addColorStop(1, "rgba(150,172,168,0)");
  ctx.fillStyle = mist;
  ctx.fillRect(0, horizon, w, h * 0.5);
  if (view.rain) {
    ctx.strokeStyle = "rgba(188,214,229,.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < 170; i++) {
      const x = (i * 317.7 + time * 0.07) % w;
      const y = (i * 93.3 + time * 0.5) % h;
      ctx.moveTo(x, y);
      ctx.lineTo(x - 4, y + 17);
    }
    ctx.stroke();
  }
  const vignette = ctx.createRadialGradient(w / 2, h / 2, h * 0.15, w / 2, h / 2, w * 0.7);
  vignette.addColorStop(0, "rgba(0,0,0,0)");
  vignette.addColorStop(1, "rgba(0,0,0,.72)");
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);
}
