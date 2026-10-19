// The landing page's 3D scenes: the hero's nine crews and the fleet cards' turntables.
//
// Ported from the "Fleet Home" design artifact (claude.ai/artifact/NNE52iufohyH8hvRQtF9qt), whose
// script ran three.js from a CDN importmap. Changes from it are kept to the module boundary: the
// bundled `three` instead of the importmap, the world data from `world.json` instead of an inline
// literal, the exports at the bottom, and code no scene calls (drawing helpers from earlier
// versions of the design) left out. The body is the artifact's, untyped; `scene.d.ts` types what
// `main.ts` calls, so the page around it is checked. Nothing here imports CesiumJS.

import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";

import WORLD from "./world.json";

// Line-art renderer for three.js: 4-tone flat shading from the anatomy palette, screen-space ink outlines
// from a normal + depth pass, detail lines, instanced primitive pools. Everything is rendered 2× supersampled
// and box-filtered down, so ink and detail lines are anti-aliased. World is Z-up, metres.

const VERT = /* glsl */ `
attribute float aLit;
attribute vec3 aGrid;
uniform float uLit;
varying vec3 vW;
varying vec3 vL;
varying float vLit;
varying vec3 vGrid;
void main() {
  vec4 p = vec4(position, 1.0);
  #ifdef USE_INSTANCING
  p = instanceMatrix * p;
  #endif
  vec4 w = modelMatrix * p;
  vW = w.xyz;
  vL = position;
  vLit = max(aLit, uLit);
  vGrid = aGrid;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const TOON_FRAG = /* glsl */ `
uniform vec3 uTop, uLitTop, uS0, uS1, uS2, uS3, uLitShade, uGridInk, uFlat;
uniform vec2 uLight;
uniform vec3 uViewDir;
uniform float uDark, uSolar, uCovOn, uFlatOn;
uniform sampler2D uCov;
uniform vec4 uCovRect;
uniform vec3 uCovInk, uBandInk;
varying vec3 vW;
varying vec3 vL;
varying float vLit;
varying vec3 vGrid;
void main() {
  vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
  if (dot(n, uViewDir) < 0.0) n = -n;
  vec3 c;
  if (n.z > 0.62) c = mix(uTop, uLitTop, vLit);
  else if (n.z < -0.62) c = uS0;
  else {
    float s = dot(normalize(n.xy + 1e-6), uLight);
    c = s > 0.5 ? uS3 : s > 0.0 ? uS2 : s > -0.5 ? uS1 : uS0;
    c = mix(c, uLitShade, vLit * 0.55);
  }
  if (uDark > 0.5) c = uS0;
  if (uFlatOn > 0.5 && n.z > 0.62) c = uFlat;
  if (uCovOn > 0.5) {
    vec2 cu = (vW.xy - uCovRect.xy) / uCovRect.zw;
    if (cu.x > 0.0 && cu.x < 1.0 && cu.y > 0.0 && cu.y < 1.0 && n.z > 0.62) {
      vec4 cv = texture2D(uCov, cu);
      c = mix(c, uCovInk, cv.r * 0.5);
      c = mix(c, uBandInk, cv.g * 0.6);
    }
  }
  // solar cells: a grid on the top face of a unit box (local coords −0.5…0.5), counts in aGrid.xy
  if (uSolar > 0.5 && vGrid.x > 0.0 && abs(vL.z) > 0.49) {
    vec2 g = (vL.xy + 0.5) * vGrid.xy;
    vec2 f = abs(fract(g) - 0.5);
    vec2 w = fwidth(g);
    float lx = smoothstep(0.5 - w.x * 1.2, 0.5, f.x), ly = smoothstep(0.5 - w.y * 1.2, 0.5, f.y);
    float fade = 1.0 - smoothstep(0.1, 0.25, max(w.x, w.y));
    c = mix(c, uGridInk, max(lx, ly) * 0.8 * fade);
  }
  gl_FragColor = vec4(c, 1.0);
}`;

const NORMAL_FRAG = /* glsl */ `
varying vec3 vW;
uniform vec3 uViewDir;
void main() {
  vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
  if (dot(n, uViewDir) < 0.0) n = -n;
  gl_FragColor = vec4(n * 0.5 + 0.5, 1.0);
}`;

const QUAD_VERT = /* glsl */ `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const EDGE_FRAG = /* glsl */ `
uniform sampler2D tN;
uniform sampler2D tD;
uniform vec2 uPx;
uniform vec3 uInk;
uniform float uRange, uMpp, uAlpha, uSil, uCrease, uTwo;
varying vec2 vUv;
vec4 N(vec2 o) { return texture2D(tN, vUv + o * uPx); }
float D(vec2 o) { return texture2D(tD, vUv + o * uPx).x * uRange; }
void main() {
  vec4 c = N(vec2(0.0));
  float dc = D(vec2(0.0));
  // silhouettes: drawn on the nearer side of a depth step; creases: from the normal change between neighbours
  float eS = 0.0, eC = 0.0;
  vec2 offs[4];
  offs[0] = vec2(1.0, 0.0); offs[1] = vec2(0.0, 1.0); offs[2] = vec2(-1.0, 0.0); offs[3] = vec2(0.0, -1.0);
  for (int i = 0; i < 4; i++) {
    vec4 s = N(offs[i]);
    float di = D(offs[i]);
    if (c.a < 0.5 && s.a < 0.5) continue;
    if ((c.a < 0.5) != (s.a < 0.5)) { eS = max(eS, c.a > 0.5 ? 1.0 : 0.0); continue; }
    if (di - dc > uMpp * 6.0 + 0.004) eS = 1.0;
    // creases one texel wide (the +x/+y neighbours only) when the supersampling is too low for two
    if (i < 2 || uTwo > 0.5) { float dn = 1.0 - dot(c.xyz * 2.0 - 1.0, s.xyz * 2.0 - 1.0); eC = max(eC, smoothstep(0.12, 0.3, dn)); }
  }
  float lap = abs(D(vec2(1.0, 0.0)) + D(vec2(-1.0, 0.0)) - 2.0 * dc) + abs(D(vec2(0.0, 1.0)) + D(vec2(0.0, -1.0)) - 2.0 * dc);
  if (c.a > 0.5 && lap > uMpp * 8.0 + 0.01) eC = max(eC, 0.8);
  gl_FragColor = vec4(uInk, max(eS * uSil, eC * uCrease) * uAlpha);
}`;
// resolve: box-filter the supersampled image down to the screen
const DOWN_FRAG = /* glsl */ `
uniform sampler2D t; uniform vec2 uSrc; uniform float uR, uK; varying vec2 vUv;
void main() {
  vec4 a = vec4(0.0);
  // well supersampled: one bilinear tap per pixel keeps lines crisp; below 1.6×, 2×2 taps near the centre smooth the stair-steps
  for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) { vec2 o = (vec2(float(i), float(j)) - 0.5) * uK; a += texture2D(t, vUv + o * uR * uSrc); }
  gl_FragColor = a / 4.0;
}`;
// FXAA (Lottes' compact form), run at the internal resolution when the supersampling budget is low
const FXAA_FRAG = /* glsl */ `
uniform sampler2D t; uniform vec2 uSrc; varying vec2 vUv;
void main() {
  vec3 L = vec3(0.299, 0.587, 0.114);
  vec3 nw = texture2D(t, vUv + vec2(-1.0, -1.0) * uSrc).rgb, ne = texture2D(t, vUv + vec2(1.0, -1.0) * uSrc).rgb;
  vec3 sw = texture2D(t, vUv + vec2(-1.0, 1.0) * uSrc).rgb, se = texture2D(t, vUv + vec2(1.0, 1.0) * uSrc).rgb;
  vec4 m = texture2D(t, vUv);
  float lNW = dot(nw, L), lNE = dot(ne, L), lSW = dot(sw, L), lSE = dot(se, L), lM = dot(m.rgb, L);
  float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE))), lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  vec2 dir = vec2(-((lNW + lNE) - (lSW + lSE)), (lNW + lSW) - (lNE + lSE));
  float red = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);
  float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + red);
  dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * uSrc;
  vec3 a = 0.5 * (texture2D(t, vUv + dir * (1.0 / 3.0 - 0.5)).rgb + texture2D(t, vUv + dir * (2.0 / 3.0 - 0.5)).rgb);
  vec3 b = a * 0.5 + 0.25 * (texture2D(t, vUv - dir * 0.5).rgb + texture2D(t, vUv + dir * 0.5).rgb);
  float lB = dot(b, L);
  gl_FragColor = vec4((lB < lMin || lB > lMax) ? a : b, 1.0);
}`;

const css = (el, name) => getComputedStyle(el).getPropertyValue(name).trim();
const col = (s) => new THREE.Color(s || "#888");

function makeEngine(canvas, { themeEl, maxPixels = 9e6 } = {}) {
  THREE.ColorManagement.enabled = false;
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  const scene = new THREE.Scene();
  // models use the anatomy convention (x east, y south, z up: left-handed). Mirror y so they render unflipped.
  scene.scale.set(1, -1, 1);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 8000);
  camera.up.set(0, 0, 1);
  const auxCam = camera.clone();

  const U = {
    uTop: { value: new THREE.Color() },
    uLitTop: { value: new THREE.Color() },
    uLitShade: { value: new THREE.Color() },
    uS0: { value: new THREE.Color() },
    uS1: { value: new THREE.Color() },
    uS2: { value: new THREE.Color() },
    uS3: { value: new THREE.Color() },
    uGridInk: { value: new THREE.Color() },
    uCov: { value: null },
    uCovRect: { value: new THREE.Vector4(0, 0, 1, 1) },
    uCovInk: { value: new THREE.Color() },
    uBandInk: { value: new THREE.Color() },
    uLight: { value: new THREE.Vector2(0.6, -0.8).normalize() },
    uViewDir: { value: new THREE.Vector3(0, 0, 1) },
  };
  U.uCov.value = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1);
  U.uCov.value.needsUpdate = true;
  const toon = (x = {}) =>
    new THREE.ShaderMaterial({
      uniforms: {
        ...U,
        uLit: { value: x.lit ? 1 : 0 },
        uDark: { value: x.dark ? 1 : 0 },
        uSolar: { value: x.solar ? 1 : 0 },
        uCovOn: { value: x.cov ? 1 : 0 },
        uFlatOn: { value: x.flat ? 1 : 0 },
        uFlat: { value: new THREE.Color() },
      },
      vertexShader: VERT,
      fragmentShader: TOON_FRAG,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
  const mats = {
    solid: toon(),
    lit: toon({ lit: true }),
    dark: toon({ dark: true }),
    solar: toon({ solar: true }),
    terrain: toon({ cov: true }),
    water: toon({ flat: "--anatomy-shade-0" }),
    road: toon({ flat: "--anatomy-shade-3" }),
    mulch: toon({ flat: "--anatomy-shade-1" }),
  };
  const flatTokens = {
    water: "--anatomy-shade-0",
    road: "--anatomy-shade-3",
    mulch: "--anatomy-shade-1",
  };
  const normalMat = new THREE.ShaderMaterial({
    uniforms: { uViewDir: U.uViewDir, uLit: { value: 0 } },
    vertexShader: VERT,
    fragmentShader: NORMAL_FRAG,
    side: THREE.DoubleSide,
  });
  const lineMats = {};
  for (const t of ["lo", "faint", "mid", "hi", "lit"])
    lineMats[t] = new THREE.LineBasicMaterial({ color: 0x888888 });
  lineMats.litDash = new THREE.LineDashedMaterial({ color: 0, dashSize: 0.14, gapSize: 0.09 });
  lineMats.zoneDash = new THREE.LineDashedMaterial({ color: 0, dashSize: 9, gapSize: 6 });
  lineMats.flight = new THREE.LineDashedMaterial({
    color: 0,
    dashSize: 10,
    gapSize: 14,
    transparent: true,
    opacity: 0.5,
  });
  const dotMats = {};
  for (const t of ["mid", "lo", "faint", "lit", "hi"])
    dotMats[t] = new THREE.PointsMaterial({ color: 0x888888, size: 2, sizeAttenuation: false });
  const tokenOf = {
    lo: "--anatomy-lo",
    faint: "--anatomy-faint",
    mid: "--anatomy-mid",
    hi: "--anatomy-hi",
    lit: "--anatomy-lit",
    litDash: "--anatomy-lit",
    zoneDash: "--anatomy-lit",
    flight: "--anatomy-mid",
  };

  const edgeMat = new THREE.ShaderMaterial({
    uniforms: {
      tN: { value: null },
      tD: { value: null },
      uPx: { value: new THREE.Vector2() },
      uInk: { value: new THREE.Color() },
      uRange: { value: 1 },
      uMpp: { value: 0.01 },
      uAlpha: { value: 0.92 },
      uSil: { value: 1 },
      uCrease: { value: 1 },
      uTwo: { value: 1 },
    },
    vertexShader: QUAD_VERT,
    fragmentShader: EDGE_FRAG,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  const quadCam = new THREE.Camera();
  const quadScene = (mat) => {
    const s = new THREE.Scene();
    s.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
    return s;
  };
  const edgeScene = quadScene(edgeMat);
  const copyMat = new THREE.ShaderMaterial({
    uniforms: { t: { value: null } },
    vertexShader: QUAD_VERT,
    fragmentShader:
      "uniform sampler2D t; varying vec2 vUv; void main(){ gl_FragColor = texture2D(t, vUv); }",
    depthTest: false,
    depthWrite: false,
  });
  const downMat = new THREE.ShaderMaterial({
    uniforms: {
      t: { value: null },
      uSrc: { value: new THREE.Vector2() },
      uR: { value: 2 },
      uK: { value: 0 },
    },
    vertexShader: QUAD_VERT,
    fragmentShader: DOWN_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const downScene = quadScene(downMat);
  const fxaaMat = new THREE.ShaderMaterial({
    uniforms: { t: { value: null }, uSrc: { value: new THREE.Vector2() } },
    vertexShader: QUAD_VERT,
    fragmentShader: FXAA_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const fxaaScene = quadScene(fxaaMat);
  const clear = new THREE.Color();

  function setTheme() {
    const el = themeEl || document.body;
    U.uTop.value.copy(col(css(el, "--anatomy-top")));
    U.uLitTop.value.copy(col(css(el, "--anatomy-lit-top")));
    U.uLitShade.value.copy(col(css(el, "--anatomy-lit-shade")));
    ["0", "1", "2", "3"].forEach((k) =>
      U["uS" + k].value.copy(col(css(el, "--anatomy-shade-" + k))),
    );
    U.uGridInk.value.copy(col(css(el, "--anatomy-lo")));
    U.uCovInk.value.copy(col(css(el, "--anatomy-shade-0")));
    U.uBandInk.value.copy(col(css(el, "--anatomy-lit-top")));
    for (const [k, tok] of Object.entries(flatTokens))
      mats[k].uniforms.uFlat.value.copy(col(css(el, tok)));
    edgeMat.uniforms.uInk.value.copy(col(css(el, "--anatomy-hi")));
    for (const [k, m] of Object.entries(lineMats)) m.color.copy(col(css(el, tokenOf[k])));
    for (const [k, m] of Object.entries(dotMats)) m.color.copy(col(css(el, "--anatomy-" + k)));
    clear.copy(col(css(el, "--anatomy-card")));
  }

  // a pass = normal/depth target + colour target at a given internal size
  function makePass(iw, ih) {
    const nrt = new THREE.WebGLRenderTarget(iw, ih, {
      type: THREE.HalfFloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    nrt.depthTexture = new THREE.DepthTexture(iw, ih, THREE.FloatType);
    const crt = new THREE.WebGLRenderTarget(iw, ih, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
    let frt = null;
    return {
      nrt,
      crt,
      iw,
      ih,
      get frt() {
        return (
          frt ||
          (frt = new THREE.WebGLRenderTarget(iw, ih, {
            minFilter: THREE.LinearFilter,
            magFilter: THREE.LinearFilter,
          }))
        );
      },
      dispose() {
        nrt.dispose();
        nrt.depthTexture.dispose();
        crt.dispose();
        if (frt) frt.dispose();
      },
    };
  }
  let W = 1,
    H = 1,
    SS = 2,
    main = null;
  const dotBase = { mid: 2.2, lo: 1.6, faint: 1.4, lit: 3, hi: 2 };
  function resize(w, h) {
    W = Math.max(1, w);
    H = Math.max(1, h);
    renderer.setSize(W, H, false);
    const dpr = renderer.getPixelRatio();
    SS = Math.max(1, Math.min(2, Math.sqrt(maxPixels / (W * H * dpr * dpr))));
    if (main) main.dispose();
    main = makePass(Math.round(W * dpr * SS), Math.round(H * dpr * SS));
    copyMat.uniforms.t.value = main.crt.texture;
    for (const [k, m] of Object.entries(dotMats)) m.size = dotBase[k] * dpr * SS;
  }

  // camera from azimuth/elevation (degrees; anatomy convention), target, view width in metres
  const view = { target: new THREE.Vector3(), az: 42, el: 32, width: 30 };
  const tmp = new THREE.Vector3();
  function aim(cam, v, aspect) {
    const a = (v.az * Math.PI) / 180,
      e = (Math.min(89.5, v.el) * Math.PI) / 180;
    const d = tmp.set(Math.cos(e) * Math.cos(a), -Math.cos(e) * Math.sin(a), Math.sin(e));
    const tg = new THREE.Vector3(v.target.x, -v.target.y, v.target.z);
    cam.position.copy(tg).addScaledVector(d, 4000);
    cam.up.set(0, 0, 1);
    cam.lookAt(tg);
    const hw = v.width / 2,
      hh = hw / aspect;
    cam.left = -hw;
    cam.right = hw;
    cam.top = hh;
    cam.bottom = -hh;
    cam.near = 1;
    cam.far = 8000;
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    U.uViewDir.value.copy(d);
    U.uLight.value.set(Math.cos(a + 0.9), -Math.sin(a + 0.9)).normalize();
  }
  // ink weight follows scale: wide views (small machines on screen) get lighter creases and outlines
  function inkFor(widthM, cssW) {
    const mpp = widthM / Math.max(1, cssW),
      far = Math.min(1, Math.max(0, (mpp - 0.012) / 0.03));
    edgeMat.uniforms.uCrease.value = 1 - 0.12 * far;
    edgeMat.uniforms.uSil.value = 1;
  }
  function runPass(pass, cam, widthM, cssW) {
    inkFor(widthM, cssW || pass.iw / 2);
    // creases two texels wide unless the supersampling is nearly gone
    edgeMat.uniforms.uTwo.value =
      pass.iw / Math.max(1, (cssW || pass.iw / 2) * renderer.getPixelRatio()) >= 1.2 ? 1 : 0;
    edgeMat.uniforms.uRange.value = cam.far - cam.near;
    edgeMat.uniforms.uMpp.value = widthM / pass.iw;
    edgeMat.uniforms.tN.value = pass.nrt.texture;
    edgeMat.uniforms.tD.value = pass.nrt.depthTexture;
    edgeMat.uniforms.uPx.value.set(1 / pass.iw, 1 / pass.ih);
    cam.layers.set(0);
    scene.overrideMaterial = normalMat;
    renderer.setRenderTarget(pass.nrt);
    renderer.setClearColor(0x000000, 0);
    renderer.clear();
    renderer.render(scene, cam);
    scene.overrideMaterial = null;
    renderer.setRenderTarget(pass.crt);
    renderer.setClearColor(clear, 1);
    cam.layers.enableAll();
    renderer.render(scene, cam);
    renderer.autoClear = false;
    renderer.render(edgeScene, quadCam);
    renderer.autoClear = true;
  }
  let overlay = null;
  function render() {
    aim(camera, view, W / H);
    runPass(main, camera, view.width, W);
    if (overlay) {
      renderer.autoClear = false;
      overlay(main.crt);
      renderer.autoClear = true;
    }
    let src = main.crt;
    if (SS < 1.25) {
      fxaaMat.uniforms.t.value = src.texture;
      fxaaMat.uniforms.uSrc.value.set(1 / main.iw, 1 / main.ih);
      renderer.setRenderTarget(main.frt);
      renderer.render(fxaaScene, quadCam);
      src = main.frt;
    }
    downMat.uniforms.t.value = src.texture;
    downMat.uniforms.uSrc.value.set(1 / main.iw, 1 / main.ih);
    downMat.uniforms.uR.value = SS;
    downMat.uniforms.uK.value = SS >= 1.6 ? 0 : 0.36;
    renderer.setRenderTarget(null);
    renderer.render(downScene, quadCam);
    aim(camera, view, W / H); // leave the uniforms set for the main view
  }
  // render another view (e.g. the magnifier) into its own pass; returns the colour texture
  const aux = new Map();
  function renderAux(key, v, cssW, cssH) {
    const dpr = renderer.getPixelRatio();
    const iw = Math.round(cssW * dpr * 2),
      ih = Math.round(cssH * dpr * 2);
    let p = aux.get(key);
    if (!p || p.iw !== iw || p.ih !== ih) {
      if (p) p.dispose();
      p = makePass(iw, ih);
      aux.set(key, p);
    }
    aim(auxCam, v, cssW / cssH);
    runPass(p, auxCam, v.width, cssW);
    aim(camera, view, W / H);
    return p.crt.texture;
  }
  function toScreen(p) {
    const v = tmp.set(p[0], -p[1], p[2] || 0).project(camera);
    return [((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H];
  }
  // lower the supersampling budget on slow GPUs (called by the app when frames run long)
  function setBudget(px) {
    maxPixels = px;
    resize(W, H);
  }
  const budget = () => maxPixels;
  return {
    THREE,
    renderer,
    scene,
    camera,
    view,
    mats,
    lineMats,
    dotMats,
    setTheme,
    resize,
    render,
    renderAux,
    toScreen,
    setBudget,
    budget,
    ss: () => SS,
    uniforms: U,
    setOverlay: (f) => (overlay = f),
    size: () => [W, H],
    quadCam,
    quadScene,
  };
}

// ── instanced primitive pools ───────────────────────────────────────────────
const unitBox = () => new THREE.BoxGeometry(1, 1, 1);
const unitCyl = (seg = 18) => {
  const g = new THREE.CylinderGeometry(1, 1, 1, seg, 1);
  g.rotateX(Math.PI / 2);
  g.translate(0, 0, 0.5);
  return g;
};
const unitCone = (seg = 18) => {
  const g = new THREE.CylinderGeometry(0, 1, 1, seg, 1);
  g.rotateX(Math.PI / 2);
  g.translate(0, 0, 0.5);
  return g;
};
const unitSph = (a = 16, b = 10) => new THREE.SphereGeometry(1, a, b);
class Pool {
  constructor(engine, geo, mat, max, parent) {
    this.lit = new THREE.InstancedBufferAttribute(new Float32Array(max), 1);
    geo.setAttribute("aLit", this.lit);
    this.grid = new THREE.InstancedBufferAttribute(new Float32Array(max * 3), 3);
    geo.setAttribute("aGrid", this.grid);
    this.mesh = new THREE.InstancedMesh(geo, mat, max);
    this.mesh.frustumCulled = false;
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.count = 0;
    this.max = max;
    this.n = 0;
    (parent || engine.scene).add(this.mesh);
  }
  begin() {
    this.n = 0;
  }
  add(m, lit = 0, grid) {
    if (this.n >= this.max) {
      if (!this.warned) {
        console.warn("pool full", this.max);
        this.warned = true;
      }
      return;
    }
    this.mesh.setMatrixAt(this.n, m);
    this.lit.array[this.n] = lit;
    if (grid) {
      this.grid.array[this.n * 3] = grid[0];
      this.grid.array[this.n * 3 + 1] = grid[1];
    }
    this.n++;
  }
  end() {
    this.mesh.count = this.n;
    this.mesh.instanceMatrix.needsUpdate = true;
    this.lit.needsUpdate = true;
    this.grid.needsUpdate = true;
  }
}

// primitives that write matrices into pools: oriented box, link, cylinder/rod/cone, sphere
function makeKit(engine, { n = 6000, parent } = {}) {
  const { mats } = engine;
  const P = {
    box: new Pool(engine, unitBox(), mats.solid, n, parent),
    rbox: new Pool(
      engine,
      new RoundedBoxGeometry(1, 1, 1, 4, 0.12),
      mats.solid,
      Math.ceil(n / 3),
      parent,
    ),
    cyl: new Pool(engine, unitCyl(), mats.solid, Math.ceil(n / 2), parent),
    cone: new Pool(engine, unitCone(), mats.solid, Math.ceil(n / 8), parent),
    sph: new Pool(engine, unitSph(), mats.solid, Math.ceil(n / 2), parent),
    dark: new Pool(engine, unitBox(), mats.dark, Math.ceil(n / 8), parent),
    solar: new Pool(engine, unitBox(), mats.solar, Math.ceil(n / 4), parent),
    mulch: new Pool(engine, unitCyl(28), mats.mulch, Math.ceil(n / 20), parent),
    disc: new Pool(engine, unitCyl(96), mats.solid, 64, parent),
  };
  const m = new THREE.Matrix4(),
    ax = new THREE.Vector3(),
    ay = new THREE.Vector3(),
    az = new THREE.Vector3();
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], (a[2] || 0) - (b[2] || 0)];
  const len = (a) => Math.hypot(a[0], a[1], a[2]);
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const unit = (a) => {
    const l = len(a) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  function obox(c, ex, ey, ez, lit = 0, pool = "box", grid) {
    m.makeBasis(
      ax.set(ex[0] * 2, ex[1] * 2, ex[2] * 2),
      ay.set(ey[0] * 2, ey[1] * 2, ey[2] * 2),
      az.set(ez[0] * 2, ez[1] * 2, ez[2] * 2),
    );
    m.setPosition(c[0], c[1], c[2] || 0);
    P[pool].add(m, lit, grid);
  }
  function link(A, B, t, lit = 0) {
    const d = sub(B, A),
      l = len(d);
    if (l < 1e-6) return;
    const ex = [d[0] / 2, d[1] / 2, d[2] / 2];
    let side = cross(ex, [0, 0, 1]);
    if (len(side) < 1e-6) side = cross(ex, [1, 0, 0]);
    const ey = unit(side).map((v) => (v * t) / 2);
    const ez = unit(cross(ex, ey)).map((v) => (v * t) / 2);
    obox([(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, ((A[2] || 0) + (B[2] || 0)) / 2], ex, ey, ez, lit);
  }
  function cyl(x, y, r, z0, h, lit = 0, pool = "cyl") {
    m.makeScale(r, r, h);
    m.setPosition(x, y, z0);
    P[pool].add(m, lit);
  }
  function cone(x, y, r, z0, h, lit = 0) {
    cyl(x, y, r, z0, h, lit, "cone");
  }
  function disc(x, y, r, z0, h, lit = 0) {
    cyl(x, y, r, z0, h, lit, "disc");
  }
  function rod(A, B, r, lit = 0) {
    const d = sub(B, A),
      l = len(d);
    if (l < 1e-6) return;
    const z = unit(d);
    let x = cross(z, [0, 0, 1]);
    if (len(x) < 1e-6) x = cross(z, [1, 0, 0]);
    x = unit(x);
    const y = cross(z, x);
    m.makeBasis(
      ax.set(x[0] * r, x[1] * r, x[2] * r),
      ay.set(y[0] * r, y[1] * r, y[2] * r),
      az.set(d[0], d[1], d[2]),
    );
    m.setPosition(A[0], A[1], A[2] || 0);
    P.cyl.add(m, lit);
  }
  function sph(c, r, lit = 0) {
    m.makeScale(r, r, r);
    m.setPosition(c[0], c[1], c[2] || 0);
    P.sph.add(m, lit);
  }
  function begin() {
    for (const k in P) P[k].begin();
  }
  function end() {
    for (const k in P) P[k].end();
  }
  function visible(v) {
    for (const k in P) P[k].mesh.visible = v;
  }
  return { pools: P, obox, link, cyl, cone, disc, rod, sph, begin, end, visible };
}

// line segments by tone (static: build once; dynamic: begin/seg/end each frame)
function makeLines(engine, { dynamic = false, max = 20000, parent } = {}) {
  const tones = ["lo", "faint", "mid", "hi", "lit", "litDash", "zoneDash", "flight"];
  const buf = {},
    objs = {};
  for (const t of tones) buf[t] = [];
  const api = {
    seg(a, b, tone = "lo") {
      buf[tone].push(a[0], a[1], a[2] || 0, b[0], b[1], b[2] || 0);
    },
    poly(pts, tone = "lo", closed = true) {
      for (let i = 0; i < pts.length - (closed ? 0 : 1); i++)
        api.seg(pts[i], pts[(i + 1) % pts.length], tone);
    },
    ring(c, r, z, tone = "lo", n = 48) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2;
        pts.push([c[0] + Math.cos(a) * r, c[1] + Math.sin(a) * r, z]);
      }
      api.poly(pts, tone);
    },
    begin() {
      for (const t of tones) buf[t].length = 0;
    },
    end() {
      for (const t of tones) {
        const arr = buf[t];
        let o = objs[t];
        if (!o) {
          if (!dynamic && !arr.length) continue;
          const g = new engine.THREE.BufferGeometry();
          const n = dynamic ? max : arr.length / 3;
          g.setAttribute("position", new engine.THREE.BufferAttribute(new Float32Array(n * 3), 3));
          if (t.endsWith("Dash") || t === "flight")
            g.setAttribute(
              "lineDistance",
              new engine.THREE.BufferAttribute(new Float32Array(n), 1),
            );
          o = objs[t] = new engine.THREE.LineSegments(g, engine.lineMats[t]);
          o.layers.set(1);
          o.frustumCulled = false;
          (parent || engine.scene).add(o);
        }
        const pos = o.geometry.attributes.position;
        const cnt = Math.min(arr.length, pos.array.length);
        pos.array.set(cnt === arr.length ? arr : arr.slice(0, cnt));
        pos.needsUpdate = true;
        const ld = o.geometry.attributes.lineDistance;
        if (ld) {
          for (let i = 0; i < cnt / 6; i++) {
            const j = i * 6;
            ld.array[i * 2] = 0;
            ld.array[i * 2 + 1] = Math.hypot(
              arr[j + 3] - arr[j],
              arr[j + 4] - arr[j + 1],
              arr[j + 5] - arr[j + 2],
            );
          }
          ld.needsUpdate = true;
        }
        o.geometry.setDrawRange(0, cnt / 3);
      }
    },
    build() {
      api.end();
      return objs;
    },
    objs,
  };
  return api;
}

// points (stipple, gravel)
function makeDots(engine, pts, tone = "mid", parent) {
  const g = new engine.THREE.BufferGeometry();
  g.setAttribute("position", new engine.THREE.Float32BufferAttribute(pts, 3));
  const p = new engine.THREE.Points(g, engine.dotMats[tone]);
  p.layers.set(1);
  p.frustumCulled = false;
  (parent || engine.scene).add(p);
  return p;
}

// True-scale 3D models (metres, Z-up) written into instanced primitive pools.
// Dimensions follow the SVG models (v2/shared/models.js) so both versions describe the same machines.
const add = (a, b) => [a[0] + b[0], a[1] + b[1], (a[2] || 0) + (b[2] || 0)];
const mul = (a, s) => [a[0] * s, a[1] * s, (a[2] || 0) * s];
const lerp3 = (a, b, u) => [
  a[0] + (b[0] - a[0]) * u,
  a[1] + (b[1] - a[1]) * u,
  a[2] + (b[2] - a[2]) * u,
];
const rot = ([x, y, z], h) => [
  x * Math.cos(h) - y * Math.sin(h),
  x * Math.sin(h) + y * Math.cos(h),
  z ?? 0,
];
const frame = (o, h) => ({ P: (p) => add(o, rot(p, h)), V: (v) => rot(v, h) });

// level of detail: set LOD.far while drawing wide views; parts smaller than a pixel or two are skipped
const LOD = { far: false };

// ── Hexapod ─────────────────────────────────────────────────────────────────
const HEX = {
  L: 0.76,
  W: 0.46,
  T: 0.13,
  zb: 0.24,
  zFold: 0.07,
  femur: 0.18,
  tibia: 0.36,
  hips: [0.28, 0, -0.28],
  foot: [0.4, 0, -0.4],
  spread: 0.5,
  arm: {
    mounts: [
      [0.27, 0.13],
      [0.27, -0.13],
    ],
    upper: 0.3,
    fore: 0.28,
    grip: 0.1,
  },
  hitch: { len: 0.24, w: 0.38, h: 0.14 },
  lid: { x0: -0.355, x1: 0.215, w: 0.42 },
};
function legIK(H, F) {
  const a = HEX.femur,
    b = HEX.tibia;
  const dx = F[0] - H[0],
    dy = F[1] - H[1];
  const d = Math.hypot(dx, dy) || 1e-6,
    dz = F[2] - H[2];
  const D = Math.min(a + b - 1e-3, Math.max(b - a + 1e-3, Math.hypot(d, dz)));
  const th =
    Math.atan2(dz, d) + Math.acos(Math.max(-1, Math.min(1, (a * a + D * D - b * b) / (2 * a * D))));
  return [
    H[0] + (dx / d) * a * Math.cos(th),
    H[1] + (dy / d) * a * Math.cos(th),
    H[2] + a * Math.sin(th),
  ];
}
function armIK(S, W, a, b) {
  const dx = W[0] - S[0],
    dy = W[1] - S[1];
  const d = Math.hypot(dx, dy) || 1e-6,
    dz = W[2] - S[2];
  const D = Math.min(a + b - 1e-3, Math.max(Math.abs(a - b) + 1e-3, Math.hypot(d, dz)));
  const th =
    Math.atan2(dz, d) + Math.acos(Math.max(-1, Math.min(1, (a * a + D * D - b * b) / (2 * a * D))));
  return [
    S[0] + (dx / d) * a * Math.cos(th),
    S[1] + (dy / d) * a * Math.cos(th),
    S[2] + a * Math.sin(th),
  ];
}
function microDrone(k, c, spin, lit) {
  k.obox(add(c, [0, 0, 0.03]), [0.06, 0, 0], [0, 0.04, 0], [0, 0, 0.022], lit);
  for (const [sx, sy] of [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ]) {
    const tip = add(c, [sx * 0.075, sy * 0.075, 0.05]);
    k.link(add(c, [0, 0, 0.04]), tip, 0.01);
  }
}
// st: heading, gait, moving, unfold 0…1, arms [{target (body-local), grip} | null ×2], payload, basketLoad, mast, lit, scan
function hexapod(k, o, st = {}) {
  const h = st.heading || 0,
    lit = st.lit ? 1 : 0;
  const { P, V } = frame(o, h);
  const u = st.unfold ?? 1;
  const zb = HEX.zFold + (HEX.zb - HEX.zFold) * u;
  const top = zb + HEX.T;
  HEX.hips.forEach((hx, i) => {
    for (const s of [-1, 1]) {
      const tri = (i + (s > 0 ? 1 : 0)) % 2;
      const ph = (st.gait || 0) + tri * Math.PI;
      const swing = st.moving ? Math.sin(ph) : 0;
      const liftF = st.moving ? Math.max(0, Math.cos(ph)) * 0.07 : 0;
      const Hl = [hx, s * (HEX.W / 2 - 0.02), zb + 0.06];
      const Fs = [HEX.foot[i] + swing * 0.09, s * HEX.spread, liftF];
      const Ks = legIK(Hl, Fs);
      const dx = hx > 0.01 ? -1 : 1;
      const Kf = [hx + dx * 0.14, s * (HEX.W / 2 - 0.04), 0.035];
      const Ff = [Kf[0] - dx * 0.27, s * (HEX.W / 2 - 0.04), 0.022];
      const H = P(Hl),
        Kn = P(lerp3(Kf, Ks, u)),
        F = P(lerp3(Ff, Fs, u));
      k.link(H, Kn, 0.045, lit);
      k.sph(Kn, 0.028);
      k.link(Kn, F, 0.032, lit);
      k.sph(F, 0.022);
    }
  });
  // body and the clear solar lid
  k.obox(
    P([0, 0, zb + HEX.T / 2]),
    V([HEX.L / 2, 0, 0]),
    V([0, HEX.W / 2, 0]),
    [0, 0, HEX.T / 2],
    lit,
    "rbox",
  );
  const Ld = HEX.lid;
  k.obox(
    P([(Ld.x0 + Ld.x1) / 2, 0, top + 0.002]),
    V([(Ld.x1 - Ld.x0) / 2, 0, 0]),
    V([0, Ld.w / 2, 0]),
    [0, 0, 0.003],
    0,
    "solar",
    [6, 3],
  );
  for (const [x, y] of [
    [0.36, 0.2],
    [0.36, -0.2],
    [-0.36, 0.2],
    [-0.36, -0.2],
  ])
    k.sph(P([x, y, zb + HEX.T * 0.62]), 0.026);
  // rear-hitch payload
  const Hh = HEX.hitch,
    x1 = -HEX.L / 2 - 0.005,
    x0 = x1 - Hh.len,
    xc = (x0 + x1) / 2,
    z0 = zb + 0.015,
    ptop = z0 + Hh.h;
  const pay = st.payload || "none";
  if (pay !== "none")
    k.obox(P([x1 + 0.003, 0, zb + 0.065]), V([0.006, 0, 0]), V([0, 0.12, 0]), [0, 0, 0.045]);
  if (pay === "basket") {
    k.obox(
      P([xc, 0, z0 + Hh.h / 2]),
      V([Hh.len / 2, 0, 0]),
      V([0, Hh.w / 2, 0]),
      [0, 0, Hh.h / 2],
      0,
      "rbox",
    );
    k.obox(
      P([xc, 0, ptop + 0.001]),
      V([Hh.len / 2 - 0.02, 0, 0]),
      V([0, Hh.w / 2 - 0.02, 0]),
      [0, 0, 0.002],
      0,
      "dark",
    );
    const n = Math.max(0, Math.min(6, Math.round(st.basketLoad ?? 6)));
    for (let i = 0; i < n; i++) {
      const p = P([xc - 0.06 + (i % 2) * 0.12, -0.12 + Math.floor(i / 2) * 0.12, ptop + 0.02]);
      k.cyl(p[0], p[1], 0.02, ptop - 0.01, 0.03);
      k.sph(add(p, [0, 0, 0.025]), 0.026);
    }
  } else if (pay === "mapping") {
    const m = P([xc, 0, 0]);
    k.obox(
      P([xc, 0, z0 + 0.045]),
      V([Hh.len / 2, 0, 0]),
      V([0, Hh.w / 2, 0]),
      [0, 0, 0.045],
      0,
      "rbox",
    );
    const mh = 0.04 + 0.5 * (st.mast ?? u);
    k.cyl(m[0], m[1], 0.016, z0 + 0.09, mh);
    k.cyl(m[0], m[1], 0.05, z0 + 0.09 + mh, 0.06, st.scan ? 1 : 0);
    k.cyl(m[0], m[1], 0.03, z0 + 0.15 + mh, 0.03);
  } else if (pay === "drone") {
    const m = P([xc, 0, 0]);
    k.obox(
      P([xc, 0, z0 + (Hh.h - 0.02) / 2]),
      V([Hh.len / 2, 0, 0]),
      V([0, Hh.w / 2, 0]),
      [0, 0, (Hh.h - 0.02) / 2],
      0,
      "rbox",
    );
    if (st.droneOn !== false)
      microDrone(k, [m[0], m[1], ptop - 0.03 + (st.droneLift || 0)], !!st.droneLift, 0);
  }
  // two arms on the front corners
  const arms = st.arms || [null, null];
  HEX.arm.mounts.forEach(([mx, my], j) => {
    const base = P([mx, my, top]);
    const sh = add(base, [0, 0, 0.05]);
    const a = arms[j];
    let W, El, tgt, psi;
    if (a && a.target) {
      tgt = P(a.target);
      W = add(tgt, [0, 0, HEX.arm.grip]);
      El = armIK(sh, W, HEX.arm.upper, HEX.arm.fore);
      psi = Math.atan2(W[1] - sh[1], W[0] - sh[0]);
    } else {
      El = add(sh, V([-HEX.arm.upper * 0.98, my * 0.18, -0.012]));
      W = add(El, V([HEX.arm.fore * 0.86, my * 0.04, 0.03]));
      tgt = add(W, V([0.07, 0, -0.01]));
      psi = h;
    }
    const grip = a ? (a.grip ?? 0) : 1;
    k.cyl(base[0], base[1], 0.05, base[2], 0.03);
    k.sph(sh, 0.032);
    k.link(sh, El, 0.042, lit);
    k.sph(El, 0.028);
    k.link(El, W, 0.034, lit);
    k.sph(W, 0.024);
    const latv = [-Math.sin(psi), Math.cos(psi), 0],
      open = 0.01 + 0.026 * (1 - grip);
    for (const s of [-1, 1])
      k.link(add(W, mul(latv, s * open)), add(tgt, mul(latv, s * open * 0.8)), 0.014, lit);
  });
}

// ── Carrier ─────────────────────────────────────────────────────────────────
const ATV = {
  L: 2.0,
  W: 1.2,
  deck: 0.62,
  pitch: 0.3,
  layers: 4,
  wheelR: 0.28,
  slotY: 0.11,
  slots: [],
};
for (let z = ATV.layers - 1; z >= 0; z--)
  for (const x of [0.72, 0.24, -0.24, -0.72]) ATV.slots.push([x, ATV.slotY, z]);
ATV.slotZ = (layer) => ATV.deck + layer * ATV.pitch;
ATV.top = ATV.deck + ATV.layers * ATV.pitch + 0.03;
// st: heading, payload (carrier|chipper|cargo|sensor), load, skip, slotPayloads, scan, lit, chip
function carrier(k, o, st = {}) {
  const h = st.heading || 0,
    lit = st.lit ? 1 : 0;
  const { P, V } = frame(o, h);
  const D0 = ATV.deck,
    wr = 0.34,
    wy = 0.74,
    tw = 0.24;
  // six big lugged tyres on a rocker-bogie each side
  for (const s of [-1, 1]) {
    for (const x of [-0.72, 0, 0.72]) {
      const c = [x, s * wy, wr];
      k.rod(P([x, s * (wy - tw / 2), wr]), P([x, s * (wy + tw / 2), wr]), wr * 0.9);
      k.rod(P([x, s * (wy + tw / 2), wr]), P([x, s * (wy + tw / 2 + 0.01), wr]), wr * 0.42);
      if (!LOD.far)
        for (let i = 0; i < 14; i++) {
          const a = (i / 14) * Math.PI * 2 + (x + s) * 0.3,
            ca = Math.cos(a),
            sa = Math.sin(a);
          k.obox(
            P([c[0] + ca * wr * 0.93, c[1], c[2] + sa * wr * 0.93]),
            V([-sa * 0.05, 0, ca * 0.05]),
            V([0, (tw / 2) * (i % 2 ? 0.9 : 0.65), 0]),
            V([ca * 0.04, 0, sa * 0.04]),
          );
        }
    }
    const pv = [0.18, s * 0.6, 0.52],
      bp = [-0.36, s * 0.6, 0.44];
    k.link(P(pv), P([0.72, s * 0.6, wr]), 0.07);
    k.link(P(pv), P(bp), 0.07);
    k.link(P(bp), P([0, s * 0.6, wr]), 0.06);
    k.link(P(bp), P([-0.72, s * 0.6, wr]), 0.06);
    k.sph(P(pv), 0.06);
    k.sph(P(bp), 0.05);
  }
  // tub, skid plate, deck
  k.obox(P([0, 0, 0.42]), V([0.92, 0, 0]), V([0, 0.5, 0]), [0, 0, 0.15], lit, "rbox");
  k.obox(P([1.04, 0, 0.36]), V([0.1, 0, -0.08]), V([0, 0.46, 0]), V([0.04, 0, 0.05]));
  k.obox(P([-1.0, 0, 0.42]), V([0.05, 0, 0]), V([0, 0.46, 0]), [0, 0, 0.12]);
  k.obox(P([0, 0, D0 - 0.03]), V([0.98, 0, 0]), V([0, 0.6, 0]), [0, 0, 0.03]);
  for (const s of [-0.34, 0.34]) k.sph(P([1.09, s, 0.48]), 0.03, 1);
  const lp = P([0.86, 0, 0]);
  k.cyl(lp[0], lp[1], 0.06, (o[2] || 0) + D0, 0.1, st.scan ? 1 : 0);
  const kind = st.payload || "carrier";
  if (kind === "carrier") {
    // roll cage: two rounded hoops, top rails and side braces; thin shelves between
    const zt = ATV.top,
      rr = 0.16,
      Y = 0.6;
    for (const x of [-0.9, 0.9]) {
      const pts = [
        [x, -Y, D0],
        [x, -Y, zt - rr],
        [x, -Y + rr * 0.3, zt - rr * 0.3],
        [x, -Y + rr, zt],
        [x, Y - rr, zt],
        [x, Y - rr * 0.3, zt - rr * 0.3],
        [x, Y, zt - rr],
        [x, Y, D0],
      ];
      for (let i = 0; i < pts.length - 1; i++) k.link(P(pts[i]), P(pts[i + 1]), 0.04);
    }
    for (const y of [-(Y - rr), Y - rr]) k.link(P([-0.9, y, zt]), P([0.9, y, zt]), 0.035);
    for (const s of [-1, 1]) {
      k.link(P([-0.9, s * Y, D0 + 0.05]), P([0.9, s * Y, ATV.slotZ(2)]), 0.025);
      k.link(P([-0.9, s * Y, zt - rr]), P([0.9, s * Y, zt - rr]), 0.03);
    }
    if (!LOD.far)
      for (let layer = 1; layer < ATV.layers; layer++)
        k.obox(P([0, 0, ATV.slotZ(layer) - 0.01]), V([0.88, 0, 0]), V([0, 0.56, 0]), [0, 0, 0.01]);
    const load = st.load ?? ATV.slots.length;
    const skip = new Set(st.skip || []);
    const pays = st.slotPayloads || [];
    ATV.slots.forEach(([x, y, layer], i) => {
      if (i < ATV.slots.length - load || skip.has(i)) return;
      if (LOD.far)
        k.obox(
          P([x, y, ATV.slotZ(layer) + HEX.zFold + HEX.T / 2]),
          V([0, HEX.L / 2 - 0.12, 0]),
          V([HEX.W / 2 - 0.02, 0, 0]),
          [0, 0, HEX.T / 2],
          0,
          "rbox",
        );
      else
        hexapod(k, P([x, y, ATV.slotZ(layer)]), {
          heading: h + Math.PI / 2,
          unfold: 0,
          payload: pays[i] || "none",
          mast: 0,
        });
    });
  } else if (kind === "chipper") {
    k.obox(P([0.05, 0, D0 + 0.28]), V([0.55, 0, 0]), V([0, 0.45, 0]), [0, 0, 0.28], lit, "rbox");
    // hopper flaring out the back, mouth dark
    const tilt = 0.25;
    k.obox(
      P([-0.72, 0, D0 + 0.33]),
      V([0.24 * Math.cos(tilt), 0, 0.24 * Math.sin(tilt)]),
      V([0, 0.3, 0]),
      [-0.22 * Math.sin(tilt), 0, 0.22 * Math.cos(tilt)],
    );
    k.obox(P([-0.955, 0, D0 + 0.28]), V([0.004, 0, 0]), V([0, 0.25, 0]), [0, 0, 0.18], 0, "dark");
    const c0 = P([0.3, 0.1, D0 + 0.56]),
      c1 = P([0.3, 0.2, D0 + 0.98]),
      c2 = P([0.3, 0.72, D0 + 1.08]);
    k.rod(c0, c1, 0.06);
    k.sph(c1, 0.07);
    k.rod(c1, c2, 0.05);
    // chips flying out of the chute onto the pile
    if (st.chip)
      for (let i = 0; i < 12; i++) {
        const f = (i * 0.137 + (st.chipPhase || 0)) % 1;
        k.sph(
          add(c2, V([Math.sin(i * 2.3) * 0.12 * f, 0.3 + 1.1 * f, -1.0 * f * f - 0.05 * f])),
          0.018,
        );
      }
    k.obox(P([0.8, 0, D0 + 0.13]), V([0.13, 0, 0]), V([0, 0.45, 0]), [0, 0, 0.13]);
  } else if (kind === "bin") {
    // open-top bin, walls 0.4 m; fill rises as plants go in
    const zb = D0,
      hw = 0.2;
    k.obox(P([0, 0, zb + 0.02]), V([0.9, 0, 0]), V([0, 0.56, 0]), [0, 0, 0.02]);
    for (const s2 of [-1, 1]) {
      k.obox(P([0, s2 * 0.55, zb + hw]), V([0.9, 0, 0]), V([0, 0.02, 0]), [0, 0, hw], lit);
      k.obox(P([s2 * 0.89, 0, zb + hw]), V([0.02, 0, 0]), V([0, 0.55, 0]), [0, 0, hw], lit);
    }
    const f = Math.max(0, Math.min(1, st.fill || 0));
    if (f > 0.01)
      k.obox(P([0, 0, zb + 0.04 + 0.16 * f]), V([0.86, 0, 0]), V([0, 0.52, 0]), [0, 0, 0.16 * f]);
  } else if (kind === "cargo") {
    k.obox(P([-0.08, 0, D0 + 0.3]), V([0.85, 0, 0]), V([0, 0.55, 0]), [0, 0, 0.3], 0, "rbox");
  } else if (kind === "sensor") {
    const m = P([0.1, 0, 0]);
    k.obox(P([0.1, 0, D0 + 0.09]), V([0.25, 0, 0]), V([0, 0.25, 0]), [0, 0, 0.09]);
    k.cyl(m[0], m[1], 0.035, D0 + 0.18, 1.1);
    k.cyl(m[0], m[1], 0.07, D0 + 1.28, 0.09, st.scan ? 1 : 0);
    k.sph([m[0], m[1], D0 + 1.46], 0.07);
  }
}

// ── Hub: 20 ft container or 16 ft trailer box ──────────────────────────────
const VARIANTS = {
  container: {
    L: 6.058,
    W: 2.438,
    H: 2.591,
    base: 0.15,
    cols: 3,
    ramp: 2.0,
    atvs: [1.85, -0.25],
    vtolPad: [-0.7, 0],
    quadPad: [1.9, 0],
    laneY: -0.56,
    frontRack: 0.6,
    chipperIn: "rack",
  },
  trailer: {
    L: 4.877,
    W: 2.438,
    H: 2.44,
    base: 0.86,
    cols: 2,
    ramp: 3.0,
    atvs: [1.26],
    vtolPad: [-0.55, 0],
    quadPad: [1.55, 0],
    laneY: -0.56,
    frontRack: 0.5,
    chipperIn: "lane",
  },
};
for (const v of Object.values(VARIANTS)) {
  v.top = v.base + v.H;
  v.floor = v.base + 0.12;
  v.mastAt = [-v.L / 2 + 0.3, -v.W / 2 + 0.3];
  v.starlinkAt = [-v.L / 2 + 0.5, v.W / 2 - 0.45];
}
const PANEL = { a: 1.722, b: 1.134 };
const rampZ = (V0, lx) => {
  const x = lx - V0.L / 2;
  if (x <= 0) return V0.floor;
  return Math.max(
    0,
    V0.floor * (1 - x / (V0.ramp * Math.cos(Math.asin(Math.min(1, V0.floor / V0.ramp))))),
  );
};
// st: variant, wings 0…1, mast 0…1, dish 0…1, doors 0…1, ramp 0…1, jacks, contents(k) callback for what's inside
function hub(k, ln, o, st = {}) {
  const V0 = VARIANTS[st.variant || "container"];
  const P = (p) => add(o, p);
  const L0 = V0.L,
    W0 = V0.W,
    t = 0.05;
  // footing: corner castings or a trailer chassis
  if (st.variant === "trailer") {
    for (const y of [-0.5, 0.5])
      k.obox(P([0, y, V0.base - 0.1]), [L0 / 2 - 0.05, 0, 0], [0, 0.06, 0], [0, 0, 0.1]);
    for (const x of [0.75, 1.65])
      for (const s of [-1, 1]) {
        k.rod(P([x, s * 0.76, 0.38]), P([x, s * 1.08, 0.38]), 0.38);
        k.rod(P([x, s * 0.75, 0.38]), P([x, s * 1.09, 0.38]), 0.2);
      }
    for (const s of [-1, 1])
      k.obox(P([1.2, s * 1.12, 0.79]), [0.8, 0, 0], [0, 0.11, 0], [0, 0, 0.015]);
    const hitch = P([-L0 / 2 - 1.5, 0, 0.55]);
    for (const y of [-0.5, 0.5]) k.link(P([-L0 / 2 + 0.1, y, V0.base - 0.12]), hitch, 0.09);
    k.cyl(hitch[0], hitch[1], 0.07, hitch[2] - 0.05, 0.1);
    const jacks = st.jacks ?? 1;
    for (const x of [-L0 / 2 + 0.25, L0 / 2 - 0.25])
      for (const s of [-1, 1]) {
        const p = P([x, s * (W0 / 2 - 0.05), 0]);
        const z1 = V0.base - 0.05,
          z0 = z1 * (1 - jacks);
        k.cyl(p[0], p[1], 0.045, z0, z1 - z0 + 0.001);
        if (jacks > 0.95) k.cyl(p[0], p[1], 0.12, 0, 0.03);
      }
  } else
    for (const [x, y] of [
      [-L0 / 2 + 0.15, -W0 / 2 + 0.15],
      [L0 / 2 - 0.15, -W0 / 2 + 0.15],
      [L0 / 2 - 0.15, W0 / 2 - 0.15],
      [-L0 / 2 + 0.15, W0 / 2 - 0.15],
    ])
      k.obox(P([x, y, V0.base / 2]), [0.15, 0, 0], [0, 0.15, 0], [0, 0, V0.base / 2]);
  // cutaway: roof off, the walls facing the viewer cut down to 0.6 m
  if (st.cut) {
    const a = ((st.viewAz ?? 42) * Math.PI) / 180,
      dv = [Math.cos(a), Math.sin(a)];
    k.obox(P([0, 0, V0.base + 0.06]), [L0 / 2, 0, 0], [0, W0 / 2, 0], [0, 0, 0.06]);
    const walls = [
      [
        [0, -1],
        [0, -(W0 / 2 - t / 2)],
        [L0 / 2, t / 2],
      ],
      [
        [0, 1],
        [0, W0 / 2 - t / 2],
        [L0 / 2, t / 2],
      ],
      [
        [-1, 0],
        [-L0 / 2 + t / 2, 0],
        [t / 2, W0 / 2],
      ],
      [
        [1, 0],
        [L0 / 2 - t / 2, 0],
        [t / 2, W0 / 2],
      ],
    ];
    for (const [n, c, hsz] of walls) {
      const near = n[0] * dv[0] + n[1] * dv[1] > 0;
      const hh = near ? 0.6 : V0.H;
      k.obox(P([c[0], c[1], V0.base + hh / 2]), [hsz[0], 0, 0], [0, hsz[1], 0], [0, 0, hh / 2]);
    }
    interior(k, null, o, V0);
    if (st.contents) st.contents(k, V0);
    return;
  }
  // shell: floor, roof, two long walls, front wall; the door end is open
  k.obox(P([0, 0, V0.base + 0.06]), [L0 / 2, 0, 0], [0, W0 / 2, 0], [0, 0, 0.06]);
  k.obox(P([0, 0, V0.top - 0.04]), [L0 / 2, 0, 0], [0, W0 / 2, 0], [0, 0, 0.04], st.lit ? 1 : 0);
  for (const s of [-1, 1])
    k.obox(
      P([0, s * (W0 / 2 - t / 2), V0.base + V0.H / 2]),
      [L0 / 2, 0, 0],
      [0, t / 2, 0],
      [0, 0, V0.H / 2],
      st.lit ? 1 : 0,
    );
  k.obox(
    P([-L0 / 2 + t / 2, 0, V0.base + V0.H / 2]),
    [t / 2, 0, 0],
    [0, W0 / 2, 0],
    [0, 0, V0.H / 2],
    st.lit ? 1 : 0,
  );
  // door-end frame
  k.obox(P([L0 / 2 - 0.04, 0, V0.top - 0.06]), [0.04, 0, 0], [0, W0 / 2, 0], [0, 0, 0.06]);
  for (const s of [-1, 1])
    k.obox(
      P([L0 / 2 - 0.04, s * (W0 / 2 - 0.05), V0.base + V0.H / 2]),
      [0.04, 0, 0],
      [0, 0.05, 0],
      [0, 0, V0.H / 2],
    );
  // corrugation, rails
  if (ln) {
    for (const s of [-1, 1])
      for (let x = -L0 / 2 + 0.3; x <= L0 / 2 - 0.25; x += 0.278)
        ln.seg(
          P([x, s * (W0 / 2 + 0.002), V0.base + 0.13]),
          P([x, s * (W0 / 2 + 0.002), V0.top - 0.12]),
          "faint",
        );
    for (let y = -W0 / 2 + 0.25; y <= W0 / 2 - 0.2; y += 0.278)
      ln.seg(
        P([-L0 / 2 - 0.002, y, V0.base + 0.13]),
        P([-L0 / 2 - 0.002, y, V0.top - 0.12]),
        "faint",
      );
    for (const z of [V0.base + 0.12, V0.top - 0.11]) {
      ln.seg(P([-L0 / 2, -W0 / 2 - 0.003, z]), P([L0 / 2, -W0 / 2 - 0.003, z]));
      ln.seg(P([-L0 / 2, W0 / 2 + 0.003, z]), P([L0 / 2, W0 / 2 + 0.003, z]));
      ln.seg(P([-L0 / 2 - 0.003, -W0 / 2, z]), P([-L0 / 2 - 0.003, W0 / 2, z]));
    }
    // roof pads
    const [vx, vy] = V0.vtolPad,
      [qx, qy] = V0.quadPad,
      zt = V0.top + 0.002;
    ln.ring(P([vx, vy]), 1.0, zt + o[2], "mid");
    ln.ring(P([vx, vy]), 0.78, zt + o[2], "mid");
    ln.ring(P([qx, qy]), 0.42, zt + o[2], "mid", 32);
    ln.seg(P([vx - 0.25, vy - 0.3, zt]), P([vx - 0.25, vy + 0.3, zt]));
    ln.seg(P([vx + 0.25, vy - 0.3, zt]), P([vx + 0.25, vy + 0.3, zt]));
    ln.seg(P([vx - 0.25, vy, zt]), P([vx + 0.25, vy, zt]));
  }
  for (let i = 0; i < 3; i++)
    k.sph(P([-L0 / 2 + 0.5 + i * 0.12, W0 / 2 + 0.01, V0.top - 0.3]), 0.025, st.online ? 1 : 0);
  // doors swing open 255°
  const doors = st.doors ?? 1;
  for (const side of [-1, 1]) {
    const hinge = [L0 / 2 + 0.03, side * (W0 / 2 - 0.02)];
    const th = (doors * 255 * Math.PI) / 180;
    const d = [Math.sin(th), -side * Math.cos(th), 0];
    const half = (W0 / 2 - 0.03) / 2;
    const c = P([hinge[0] + d[0] * half, hinge[1] + d[1] * half, V0.base + V0.H / 2]);
    k.obox(c, mul(d, half), mul([-d[1], d[0], 0], 0.025), [0, 0, V0.H / 2 - 0.04]);
    if (ln)
      for (const f of [0.32, 0.72]) {
        for (const sgn of [-1, 1]) {
          const p = add(
            P([hinge[0], hinge[1], 0]),
            add(mul(d, half * 2 * f), mul([-d[1], d[0], 0], sgn * 0.027)),
          );
          ln.seg([p[0], p[1], V0.base + 0.12 + o[2]], [p[0], p[1], V0.top - 0.12 + o[2]]);
        }
      }
  }
  // ramp, in the carrier lane
  const down = st.ramp ?? 1;
  if (doors > 0.6) {
    const len0 = V0.ramp,
      rise = V0.floor;
    const end = Math.asin(Math.min(1, rise / len0));
    const tilt = Math.PI / 2 - (Math.PI / 2 + end) * down;
    const along = [Math.cos(tilt), 0, Math.sin(tilt)];
    const sill = [L0 / 2 - 0.02, V0.laneY, rise];
    const c = add(sill, mul(along, len0 / 2));
    const nrm = [-Math.sin(tilt), 0, Math.cos(tilt)];
    k.obox(P(add(c, mul(nrm, -0.025))), mul(along, len0 / 2), [0, 0.68, 0], mul(nrm, 0.025));
    if (ln && down > 0.99)
      for (let s = 0.2; s < len0 - 0.1; s += 0.22) {
        const p = add(sill, mul(along, s));
        ln.seg(P(add(p, [0, -0.62, 0.003])), P(add(p, [0, 0.62, 0.003])), "faint");
      }
  }
  // wings: top-hinged on both long edges; u 0 = hanging over the wall, 1 = 8° below horizontal
  const u = st.wings ?? 1;
  for (const s of [-1, 1]) {
    const ang = ((-90 + 82 * u) * Math.PI) / 180;
    const d = [0, s * Math.cos(ang), Math.sin(ang)];
    const hinge = [0, s * (W0 / 2 + 0.04), V0.top - 0.04];
    let n = [0, -d[2] * s, d[1] * s]; // perpendicular in the y-z plane
    if (u > 0.5 ? n[2] < 0 : n[1] * s < 0) n = mul(n, -1);
    const width = V0.cols * PANEL.a + 0.08,
      depthW = 2 * PANEL.b + 0.06;
    const fc = add(hinge, mul(d, depthW / 2));
    k.obox(P(add(fc, mul(n, -0.035))), [width / 2, 0, 0], mul(d, depthW / 2), mul(n, 0.025));
    for (let r = 0; r < 2; r++)
      for (let c = 0; c < V0.cols; c++) {
        const pc = add(
          add([-width / 2 + 0.04 + PANEL.a * (c + 0.5), 0, 0], hinge),
          mul(d, 0.03 + PANEL.b * (r + 0.5)),
        );
        k.obox(
          P(pc),
          [PANEL.a / 2 - 0.012, 0, 0],
          mul(d, PANEL.b / 2 - 0.012),
          mul(n, 0.01),
          0,
          "solar",
          [6, 2],
        );
      }
    if (u > 0.4)
      for (const x of [-width / 2 + 0.5, width / 2 - 0.5]) {
        const a = [x, s * (W0 / 2 + 0.01), V0.base + V0.H * 0.45];
        const b = add([x, 0, 0], add(hinge, add(mul(d, depthW * 0.62), mul(n, -0.06))));
        k.link(P(a), P(lerp3(a, b, Math.min(1, (u - 0.4) / 0.6))), 0.04);
      }
  }
  // mast 0.9 → 3 m, radio, whips
  const ext = st.mast ?? 1;
  {
    const [x, y] = V0.mastAt,
      z0 = V0.top;
    const total = 0.9 + 2.1 * ext;
    const p = P([x, y, 0]);
    k.cyl(p[0], p[1], 0.12, z0, 0.07);
    let z = z0,
      left = total;
    [0.06, 0.047, 0.036].forEach((r, i) => {
      const hh = Math.min(1.0, Math.max(0, left));
      if (hh <= 0.01 && i) return;
      k.cyl(p[0], p[1], r, z, Math.max(0.02, hh));
      z += hh - 0.12;
      left -= hh - 0.12;
    });
    const topZ = z0 + total;
    k.obox([p[0], p[1], topZ - 0.25], [0, 0.38, 0], [0.022, 0, 0], [0, 0, 0.022]);
    for (const dy of [-0.35, 0.35]) {
      k.cyl(p[0], p[1] + dy, 0.011, topZ - 0.23, 0.62);
      k.sph([p[0], p[1] + dy, topZ + 0.39], 0.012, 1);
    }
    k.obox([p[0] + 0.08, p[1], topZ - 0.58], [0.04, 0, 0], [0, 0.1, 0], [0, 0, 0.12]);
  }
  // Starlink
  {
    const [x, y] = V0.starlinkAt,
      z0 = V0.top,
      up = st.dish ?? 1;
    const tilt = ((10 + 55 * up) * Math.PI) / 180;
    const p = P([x, y, 0]);
    k.cyl(p[0], p[1], 0.025, z0, 0.32);
    const ex = [-Math.cos(tilt) * 0.192, 0, Math.sin(tilt) * 0.192];
    k.obox(
      [p[0], p[1], z0 + 0.36],
      ex,
      [0, 0.297, 0],
      mul([Math.sin(tilt), 0, Math.cos(tilt)], 0.02),
    );
  }
  // interior
  interior(k, ln, o, V0);
  if (st.contents) st.contents(k, V0);
}
function interior(k, ln, o, V0) {
  const P = (p) => add(o, p);
  const t = 0.05,
    z0 = V0.floor,
    zc = V0.top - 0.1,
    xi0 = -V0.L / 2 + t,
    xi1 = V0.L / 2 - t,
    yi1 = V0.W / 2 - t;
  // power + compute rack across the front wall
  const rw = V0.frontRack,
    face = xi0 + rw;
  k.obox(
    P([xi0 + rw / 2, 0, (z0 + zc) / 2]),
    [rw / 2, 0, 0],
    [0, V0.W / 2 - t, 0],
    [0, 0, (zc - z0) / 2],
  );
  if (ln) {
    for (const z of [0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
      if (z0 + z < zc - 0.1)
        ln.seg(
          P([face + 0.003, -V0.W / 2 + 0.1, z0 + z]),
          P([face + 0.003, V0.W / 2 - 0.1, z0 + z]),
        );
  }
  for (const z of [0.45, 0.75, 1.05]) k.sph(P([face + 0.01, -V0.W / 2 + 0.2, z0 + z]), 0.02, 1);
  // side rack beside the lane
  const ry0 = V0.laneY + 0.62,
    rx0 = face + 0.04,
    rx1 = xi1 - 0.04,
    rd = yi1 - ry0;
  let x = rx0 + 0.04;
  if (V0.chipperIn === "rack") {
    chipperModule(k, P([x + 1.06, ry0 + 0.5, z0]));
    x += 1.66;
  }
  const cw = Math.min(1.7, rx1 - x - 0.9);
  k.obox(
    P([x + cw / 2, ry0 + 0.5, z0 + 0.3]),
    [cw / 2, 0, 0],
    [0, 0.47, 0],
    [0, 0, 0.3],
    0,
    "rbox",
  );
  x += cw + 0.06;
  if (x < rx1 - 0.6) {
    const c = P([x + 0.25, ry0 + 0.5, 0]);
    k.obox([c[0], c[1], z0 + 0.09], [0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.09]);
    k.cyl(c[0], c[1], 0.035, z0 + 0.18, 0.75);
    k.cyl(c[0], c[1], 0.07, z0 + 0.93, 0.09);
  }
  const levels = [z0 + 1.2, z0 + 1.58, z0 + 1.96].filter((z) => z < zc - 0.2);
  levels.forEach((zl, li) => {
    k.obox(
      P([(rx0 + rx1) / 2, ry0 + rd / 2, zl - 0.015]),
      [(rx1 - rx0) / 2, 0, 0],
      [0, rd / 2, 0],
      [0, 0, 0.015],
    );
    for (let xx = rx0 + 0.04, j = 0; xx + 0.26 < rx1; xx += 0.3, j++) {
      const pack = li === 1 && j % 3 === 2;
      for (const yy of [ry0 + 0.06, ry0 + 0.52]) {
        const hgt = pack ? 0.2 : 0.14;
        k.obox(
          P([xx + 0.12, yy + (pack ? 0.2 : 0.19), zl + hgt / 2]),
          [0.12, 0, 0],
          [0, pack ? 0.2 : 0.19, 0],
          [0, 0, hgt / 2],
          0,
          "rbox",
        );
      }
    }
  });
  for (let i = 0; i <= 4; i++) {
    const xx = rx0 + ((rx1 - rx0) * i) / 4;
    k.obox(P([xx, ry0, (z0 + zc) / 2]), [0.02, 0, 0], [0, 0.02, 0], [0, 0, (zc - z0) / 2]);
  }
  if (V0.chipperIn === "lane") chipperModule(k, P([face + 1.0, V0.laneY, z0]));
}
function chipperModule(k, c) {
  k.obox([c[0], c[1], c[2] + 0.28], [0.55, 0, 0], [0, 0.45, 0], [0, 0, 0.28], 0, "rbox");
  const tilt = 0.25;
  k.obox(
    [c[0] - 0.77, c[1], c[2] + 0.33],
    [0.24 * Math.cos(tilt), 0, 0.24 * Math.sin(tilt)],
    [0, 0.3, 0],
    [-0.22 * Math.sin(tilt), 0, 0.22 * Math.cos(tilt)],
  );
}

// ── Drones, person, broom ───────────────────────────────────────────────────
function vtol(k, o, st = {}, ln) {
  const h = st.heading || 0,
    lit = st.lit ? 1 : 0;
  const { P, V } = frame(o, h);
  k.obox(P([-0.015, 0, 0.2]), V([0.665, 0, 0]), V([0, 0.13, 0]), [0, 0, 0.1], lit, "rbox");
  const g = P([-0.2, 0, 0]);
  k.cyl(g[0], g[1], 0.045, o[2] + 0.3, 0.02);
  k.sph(P([0.5, 0, 0.075]), 0.055);
  for (const s of [-1, 1]) {
    k.obox(P([-0.07, s * 0.66, 0.26]), V([0.145, 0, 0]), V([0, 0.535, 0]), [0, 0, 0.018], lit);
    k.obox(P([0.04, s * 0.55, 0.235]), V([0.82, 0, 0]), V([0, 0.03, 0]), [0, 0, 0.025]);
    k.obox(P([-0.7, s * 0.55, 0.36]), V([0.09, 0, 0]), V([0, 0.008, 0]), [0, 0, 0.1]);
    for (const x of [-0.62, 0.72]) {
      const m = P([x, s * 0.55, 0]);
      k.cyl(m[0], m[1], 0.045, o[2] + 0.26, 0.07);
      if (ln) ln.ring(m, 0.33, o[2] + 0.335, "mid", 28);
    }
    const lg = P([0, s * 0.13, 0]);
    k.obox([lg[0], lg[1], o[2] + 0.008], V([0.3, 0, 0]), V([0, 0.008, 0]), [0, 0, 0.008]);
    k.link([lg[0], lg[1], o[2] + 0.01], [lg[0], lg[1], o[2] + 0.11], 0.016);
  }
  k.rod(P([-0.73, 0, 0.2]), P([-0.7, 0, 0.2]), 0.17);
}
function quad(k, o, st = {}, ln) {
  const h = st.heading || 0,
    lit = st.lit ? 1 : 0;
  const { P, V } = frame(o, h);
  k.obox(P([0, 0, 0.0625]), V([0.11, 0, 0]), V([0, 0.0475, 0]), [0, 0, 0.0325], lit, "rbox");
  k.sph(P([0.12, 0, 0.03]), 0.025);
  for (const [sx, sy] of [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ]) {
    const tip = P([sx * 0.125, sy * 0.13, 0.09]);
    k.link(P([0, 0, 0.07]), tip, 0.016);
    k.cyl(tip[0], tip[1], 0.016, tip[2] - 0.015, 0.03);
    if (ln) ln.ring(tip, 0.12, tip[2] + 0.018, "mid", 20);
  }
}
function person(k, o, h = 0) {
  const { P, V } = frame(o, h);
  for (const s of [-1, 1]) {
    const f = P([0, s * 0.1, 0]);
    k.cyl(f[0], f[1], 0.075, o[2], 0.88);
  }
  k.obox(P([0, 0, 1.17]), V([0.12, 0, 0]), V([0, 0.21, 0]), [0, 0, 0.31], 0, "rbox");
  for (const s of [-1, 1]) k.link(P([0.02, s * 0.25, 1.44]), P([0.06, s * 0.27, 0.86]), 0.08);
  const nk = P([0, 0, 0]);
  k.cyl(nk[0], nk[1], 0.05, o[2] + 1.48, 0.08);
  k.sph(P([0, 0, 1.68]), 0.115);
}
// broom: a clump of lobes; tip lays it over along dir, lift raises it, shrink feeds it away.
// kind: broom / brush / weed (lobes, scaled), thistle (stems + spiny heads), pampas (blades + plumes)
function shrub(k, o, st = {}) {
  const seed = st.seed || 1,
    sc = (st.scale || 1) * (1 - (st.shrink || 0) * 0.85),
    tip = st.tip || 0,
    lift = st.lift || 0,
    dir = st.dir || [1, 0];
  const ang = tip * 1.35,
    lit = st.lit ? 1 : 0,
    kind = st.kind || "broom";
  const tf = ([x, y, z]) => {
    const along = x * dir[0] + y * dir[1],
      side = -x * dir[1] + y * dir[0];
    const a2 = along * Math.cos(ang) + z * Math.sin(ang),
      z2 = -along * Math.sin(ang) + z * Math.cos(ang);
    return add(o, [a2 * dir[0] - side * dir[1], a2 * dir[1] + side * dir[0], z2 + lift]);
  };
  const base = tf([0, 0, 0]);
  if (kind === "pampas") {
    for (let i = 0; i < 13; i++) {
      const a = seed * 1.3 + i * 2.39,
        r = 0.42 + 0.16 * Math.sin(i * 1.7 + seed);
      k.link(
        tf([Math.cos(a) * 0.06 * sc, Math.sin(a) * 0.06 * sc, 0.05 * sc]),
        tf([Math.cos(a) * r * sc, Math.sin(a) * r * sc, (0.62 + 0.2 * Math.cos(i * 2.1)) * sc]),
        0.035 * sc,
      );
    }
    k.sph(tf([0, 0, 0.14 * sc]), 0.17 * sc);
    for (let i = 0; i < 3; i++) {
      const a = seed + i * 2.1,
        top = tf([Math.cos(a) * 0.16 * sc, Math.sin(a) * 0.16 * sc, (1.55 + 0.12 * i) * sc]);
      k.link(base, top, 0.018 * sc);
      const pl = tf([Math.cos(a) * 0.17 * sc, Math.sin(a) * 0.17 * sc, (1.72 + 0.12 * i) * sc]);
      k.link(top, pl, 0.09 * sc, lit);
    }
    return;
  }
  if (kind === "thistle") {
    for (let i = 0; i < 4; i++) {
      const a = seed * 2 + i * 1.57;
      k.sph(tf([Math.cos(a) * 0.13 * sc, Math.sin(a) * 0.13 * sc, 0.07 * sc]), 0.12 * sc);
    }
    for (let i = 0; i < 6; i++) {
      const a = seed * 1.1 + i * 2.39,
        r = (0.08 + 0.1 * (i % 3)) * sc,
        z = (0.62 + 0.14 * Math.sin(i * 1.9)) * sc;
      const h = tf([Math.cos(a) * r, Math.sin(a) * r, z]);
      k.link(base, h, 0.02 * sc);
      k.sph(h, 0.055 * sc, lit);
      for (const d of [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ])
        k.link(
          h,
          tf([
            Math.cos(a) * r + d[0] * 0.07 * sc,
            Math.sin(a) * r + d[1] * 0.07 * sc,
            z - 0.02 * sc,
          ]),
          0.008,
        );
    }
    return;
  }
  for (const s of [
    [0.05, 0.03, 0.6],
    [-0.04, 0.05, 0.7],
    [0.02, -0.06, 0.65],
  ])
    k.link(base, tf(mul(s, sc)), 0.03);
  for (let i = 0; i < 7; i++) {
    const a = seed * 2.4 + i * 2.39,
      ring = i === 6 ? 0 : i < 3 ? 0.3 : 0.17,
      r = (i === 6 ? 0.27 : 0.25) + 0.04 * Math.sin(seed * 3 + i);
    k.sph(
      tf([
        Math.cos(a) * ring * sc,
        Math.sin(a) * ring * sc,
        (i === 6 ? 0.98 : i < 3 ? 0.42 : 0.72) * sc,
      ]),
      r * sc,
      lit,
    );
  }
}

// ── Plants, ground props ────────────────────────────────────────────────────
function sapling(k, o, grow = 1) {
  const g = grow;
  k.cyl(o[0] + 0.08, o[1], 0.008, o[2], 0.55 * g);
  k.cyl(o[0], o[1], 0.055, o[2], 0.42 * g);
  k.cyl(o[0], o[1], 0.01, o[2] + 0.4 * g, 0.2 * g);
  for (const [x, y, z, r] of [
    [0, 0, 0.66, 0.085],
    [0.06, 0.03, 0.58, 0.06],
    [-0.05, -0.04, 0.6, 0.06],
  ])
    k.sph([o[0] + x * g, o[1] + y * g, o[2] + z * g], r * g);
}
function mulchRing(k, o, r) {
  k.cyl(o[0], o[1], r, o[2] + 0.002, 0.012, 0, "mulch");
}
// conifer seedling, ~0.4 m
function conifer(k, o, g = 1) {
  k.cyl(o[0], o[1], 0.014, o[2], 0.14 * g);
  k.cone(o[0], o[1], 0.15 * g, o[2] + 0.07 * g, 0.24 * g);
  k.cone(o[0], o[1], 0.1 * g, o[2] + 0.22 * g, 0.2 * g);
  k.cyl(o[0] + 0.12, o[1], 0.007, o[2], 0.3 * g);
}
// native bunchgrass plug
function plug(k, o, g = 1, seed = 1) {
  for (let i = 0; i < 9; i++) {
    const a = seed + i * 2.39,
      r = 0.1 + 0.05 * Math.sin(i * 1.3);
    k.link(
      [o[0] + Math.cos(a) * 0.02, o[1] + Math.sin(a) * 0.02, o[2]],
      [
        o[0] + Math.cos(a) * r * g,
        o[1] + Math.sin(a) * r * g,
        o[2] + (0.26 + 0.06 * Math.cos(i)) * g,
      ],
      0.014,
    );
  }
}
// seeded patch: sprouts scattered over ~0.5 m
function seedPatch(k, o, g = 1, seed = 1) {
  for (let i = 0; i < 22; i++) {
    const a = seed * 3.1 + i * 2.39996,
      r = 0.44 * Math.sqrt((i + 0.5) / 22);
    const p = [o[0] + Math.cos(a) * r, o[1] + Math.sin(a) * r, o[2]];
    if (g < 0.5) k.sph([p[0], p[1], p[2] + 0.01], 0.014);
    else {
      const q = [
        p[0] + 0.02 * Math.cos(a * 3),
        p[1] + 0.02 * Math.sin(a * 3),
        p[2] + (0.08 + 0.05 * Math.sin(i)) * g,
      ];
      k.link(p, q, 0.012);
      k.sph(q, 0.02);
    }
  }
}
// straw wattle along heading h, staked
function wattle(k, o, len, h, g = 1) {
  const d = [Math.cos(h), Math.sin(h)],
    L = len * g;
  if (L < 0.05) return;
  k.rod(
    [o[0] - (d[0] * L) / 2, o[1] - (d[1] * L) / 2, o[2] + 0.1],
    [o[0] + (d[0] * L) / 2, o[1] + (d[1] * L) / 2, o[2] + 0.1],
    0.1,
  );
  if (g > 0.95)
    for (const f of [-0.32, 0.32])
      k.rod(
        [o[0] + d[0] * len * f - d[1] * 0.1, o[1] + d[1] * len * f + d[0] * 0.1, o[2]],
        [o[0] + d[0] * len * f - d[1] * 0.1, o[1] + d[1] * len * f + d[0] * 0.1, o[2] + 0.3],
        0.018,
      );
}
// distribution pole: 10.8 m, crossarm across the line, three insulators
function pole(k, o, n) {
  k.cyl(o[0], o[1], 0.15, o[2], 10.8);
  const z = o[2] + 10.15;
  k.obox(
    [o[0], o[1], z],
    [n[0] * 1.25, n[1] * 1.25, 0],
    [-n[1] * 0.06, n[0] * 0.06, 0],
    [0, 0, 0.06],
  );
  for (const f of [-1.05, 0, 1.05]) {
    const p = [o[0] + n[0] * f + (f ? 0 : -n[1] * 0.12), o[1] + n[1] * f + (f ? 0 : n[0] * 0.12)];
    k.cyl(p[0], p[1], 0.05, z + 0.06, f ? 0.22 : 0.62, 1);
  }
  k.cyl(o[0] - n[1] * 0.42, o[1] + n[0] * 0.42, 0.28, o[2] + 7.6, 0.85);
}
const POLE_TIPS = (o, n) =>
  [-1.05, 0, 1.05].map((f) => [
    o[0] + n[0] * f + (f ? 0 : -n[1] * 0.12),
    o[1] + n[1] * f + (f ? 0 : n[0] * 0.12),
    o[2] + 10.15 + 0.06 + (f ? 0.22 : 0.62),
  ]);
function oak(k, o, s) {
  k.cyl(o[0], o[1], 0.38 * s, o[2], 3.6 * s);
  for (const [dx, dy, dz, r] of [
    [0, 0, 6.4, 4.6],
    [1.9, -1.2, 5.2, 3.3],
  ])
    k.sph([o[0] + dx * s, o[1] + dy * s, o[2] + dz * s], r * s);
}
function willow(k, o, s) {
  k.cyl(o[0], o[1], 0.3 * s, o[2], 2.6 * s);
  for (const [dx, dy, dz, r] of [
    [-1.1, 0.6, 4.4, 2.8],
    [0.6, -0.5, 5.6, 3.1],
  ])
    k.sph([o[0] + dx * s, o[1] + dy * s, o[2] + dz * s], r * s);
}
function fir(k, o, s) {
  k.cyl(o[0], o[1], 0.5 * s, o[2], 4 * s);
  k.cone(o[0], o[1], 4.6 * s, o[2] + 3.5 * s, 11.5 * s);
  k.cone(o[0], o[1], 3.6 * s, o[2] + 9 * s, 9 * s);
}
function rock(k, o, r) {
  k.obox(
    [o[0], o[1], o[2] + r * 0.25],
    [r * 1.1, 0, 0],
    [0, r * 0.9, 0],
    [0, 0, r * 0.25],
    0,
    "rbox",
  );
}

// Seven scenarios on the same parcel and hub. Each one: a problem (what the scan flags), the robotic job
// (targets pulled to a chipper or a bin, items put in), and a visible outcome. Cell-relative coords are metres
// from Wd.CELL (x east, y south, z up).
function makeScenarios(Wd) {
  const C = Wd.CELL;
  const cl = (x, y) => [C[0] + x, C[1] + y];
  const rng = (s) => () => (s = (s * 16807) % 2147483647) / 2147483647;
  const f2 = (v) => Math.round(v * 100) / 100;
  const blob = (cx, cy, rx, ry, wob, ph, n = 48) =>
    Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2;
      const r = 1 + wob * (Math.sin(a * 3 + ph) * 0.6 + Math.sin(a * 5 + ph * 1.7) * 0.4);
      return [f2(cx + Math.cos(a) * rx * r), f2(cy + Math.sin(a) * ry * r)];
    });
  const rect = (x0, y0, x1, y1) => [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
  const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  // n points in a cell-relative box, spaced, passing `ok`; returned in world coords
  function scatter(seed, n, minD, box, ok, kind, scale) {
    const r = rng(seed),
      out = [];
    for (let i = 0; i < 8000 && out.length < n; i++) {
      const p = [box[0] + r() * (box[2] - box[0]), box[1] + r() * (box[3] - box[1])];
      if (!ok(p) || out.some((q) => d2(q, p) < minD)) continue;
      out.push(p);
    }
    return out.map(([x, y], i) => ({
      x: f2(C[0] + x),
      y: f2(C[1] + y),
      seed: 1 + i * 1.7,
      scale: scale[0] + (scale[1] - scale[0]) * r(),
      kind,
    }));
  }
  const grid = (xs, ys, ok = () => true) =>
    xs
      .flatMap((x) => ys.map((y) => [x, y]))
      .filter(ok)
      .map(([x, y]) => cl(x, y));
  const away = (p, list, d) => list.every((q) => d2(p, q) > d);
  const DEF = {
    atv: [1, 5],
    atvH: Math.PI,
    chip: [5, -0.6],
    chipH: 0,
    pile: [5.4, 1.5],
    via: [
      [8.5, -3.5],
      [2.2, -2.4],
    ],
  };
  const keepOut = (lay) => [lay.atv, lay.chip, lay.pile, [lay.atv[0], lay.atv[1] - 2]];

  // ── infrastructure: a distribution line crossing the cell on a diagonal ──
  const POLE = [5.6, 6.2],
    LDIR = [-Math.SQRT1_2, -Math.SQRT1_2],
    LN = [Math.SQRT1_2, -Math.SQRT1_2];
  const lineDist = (p) => Math.abs((p[0] - POLE[0]) * LN[0] + (p[1] - POLE[1]) * LN[1]);
  const infraLay = {
    atv: [-1.6, 5.6],
    atvH: Math.PI,
    chip: [5.2, -1.6],
    chipH: 0,
    pile: [5.6, 0.5],
    via: [
      [9, -4.5],
      [2.6, -3.4],
    ],
  };
  const poles = []; // world positions along the line, 60 m spans, inside the parcel
  for (let s = -1200; s <= 1200; s += 60) {
    const p = cl(POLE[0] - LDIR[0] * s, POLE[1] - LDIR[1] * s);
    if (
      p[0] > Wd.LAND.x + 8 &&
      p[0] < Wd.LAND.x + Wd.LAND.w - 8 &&
      p[1] > Wd.LAND.y + 8 &&
      p[1] < Wd.LAND.y + Wd.LAND.d - 8
    )
      poles.push(p);
  }
  poles.sort((a, b) => a[0] - b[0]);
  const corridor = (w) => {
    const a = poles[0],
      b = poles[poles.length - 1];
    const n = [LN[0] * w, LN[1] * w];
    return [
      [a[0] + n[0], a[1] + n[1]],
      [b[0] + n[0], b[1] + n[1]],
      [b[0] - n[0], b[1] - n[1]],
      [a[0] - n[0], a[1] - n[1]],
    ];
  };

  // ── agriculture: vine rows along x, 2.4 m apart; rows end at x = 3.4 (headland to the east) ──
  const ROWS = [-6.0, -3.6, -1.2, 1.2, 3.6, 6.0];
  const agriLay = {
    atv: [6.6, 2.2],
    atvH: Math.PI / 2,
    chip: [6.6, -3.2],
    chipH: Math.PI / 2,
    pile: [8, -3],
    via: [
      [12, -9],
      [6.6, -7.4],
    ],
    atvVia: [
      [12, -9],
      [8.4, -5.6],
      [6.6, -2.8],
    ],
  };
  const VINE_X = (x) => Math.abs(((x + 30) % 1.8) - 0.9) > 0.25;

  // ── fuels: a cabin, an oak beside it, shrubs under the canopy ──
  const HOUSE = { c: [-8.6, 5.4], w: 7.2, d: 6.2, h: 2.8 };
  const OAK = [-0.6, 0.4];
  const houses = [cl(...HOUSE.c), [-176, 236], [-118, 312], [-300, 268], [-352, 108], [-214, 80]];

  // ── rangeland: fence on the far side, a round trough ──
  const FENCE_Y = -5.4,
    TROUGH = [-4.9, -2.4];

  // ── reclamation: quarry benches stepping up to the far (−y) side ──
  const BENCH = (k) => ({
    ring: rect(C[0] - 130, C[1] - 95, C[0] + 70, C[1] - 6.8 - k * 4),
    z: k * 3.6,
    h: 3.6,
  });

  // ── solar: fixed-tilt panel rows along x, 7.2 m pitch, low edge facing south (+y); rows end at x = 3.4 ──
  const PV = [-10.8, -3.6, 3.6, 10.8],
    PV_X0 = -30,
    PV_X1 = 3.4;
  const pvPosts = PV.flatMap((y) => {
    const o = [];
    for (let x = PV_X0; x <= PV_X1; x += 4.3)
      for (const dy of [-0.85, 0.85]) o.push([...cl(x, y + dy), 0.12]);
    return o;
  });

  // ── pipeline: buried line along x at y = −3.5, a 15 m right-of-way, a fenced valve site on the line ──
  const PIPE_Y = -3.5,
    VALVE = { x0: -13.5, x1: -8.5, y0: -6.2, y1: -0.8 };

  const S = {
    nature: {
      id: "nature",
      chip: "Nature",
      title: "Nature · broom to native oak",
      problem:
        "French broom has taken 2 ha of oak woodland. It crowds out seedlings and burns hot. Pulling by hand is slow and the seed bank lasts for decades.",
      solve:
        "The scan maps every plant. Hexapods pull broom roots and all, chip it on site, and plant native oaks where it stood.",
      outcome:
        "A cleared, replanted cell every few minutes, logged plant by plant, with follow-up visits to catch seedlings as they sprout.",
      zone: [Wd.ZONE_R],
      zoneLabel: "Broom patch to restore",
      zoneAt: Wd.PATCH.c,
      flagLabel: "Broom flagged for removal",
      flagged: "2.1 ha of broom flagged",
      readout: "broom → native oaks",
      targets: Wd.TARGETS.map((t) => ({ ...t, kind: "broom" })),
      dest: "chipper",
      plant: { item: "oak", spots: "targets", planters: 2 },
      layout: DEF,
      view: { c: [1.6, 1.0, 0.7], w: 13 },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the broom`,
        problem: "Broom is invasive: it crowds out oaks and burns hot",
        pull: "Pulled out, roots and all",
        feed: "Chipped on site",
        plant: "A native oak goes in where the broom was",
      },
      aria: "invasive broom pulled, chipped on site and replaced with native oaks",
    },
    infra: {
      id: "infra",
      chip: "Infrastructure",
      title: "Infrastructure · power-line corridor",
      problem:
        "Brush grows up under distribution lines. Crews and mowers must keep it clear, and every pole needs inspecting. Both mean trucks and people on the line.",
      solve:
        "The scan maps the corridor and every pole. Hexapods pull the brush and chip it, and low native grasses go in. The quadcopter takes lidar and thermal of the pole hardware.",
      outcome:
        "A corridor kept clear of tall growth, with a pole-by-pole record of clearance and hardware condition.",
      zone: [corridor(16)],
      zoneLabel: "Line corridor, 32 m wide",
      zoneAt: cl(-60, -60),
      flagLabel: "Brush flagged under the line",
      flagged: "brush flagged under 1.2 km of line",
      readout: "brush → low grasses · pole checked",
      targets: scatter(
        7,
        8,
        1.7,
        [-4.6, -4.2, 3.8, 3.8],
        (p) => lineDist(p) < 2.2 && away(p, keepOut(infraLay), 2.0) && d2(p, POLE) > 2.5,
        "brush",
        [0.66, 0.8],
      ),
      dest: "chipper",
      plant: { item: "plug", spots: "targets", planters: 2 },
      layout: infraLay,
      view: { c: [1.4, 1.6, 4.4], w: 22 },
      quad: (t) => {
        const top = cl(POLE[0] + LN[0] * 2.4, POLE[1] + LN[1] * 2.4);
        return [top[0] + Math.sin(t * 0.6) * 0.25, top[1] + Math.cos(t * 0.6) * 0.25, 10.3];
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the line`,
        problem: "Brush under the line: a fire and outage risk",
        pull: "Pulled out, roots and all",
        feed: "Chipped on site",
        plant: "Low native grasses go in: nothing tall grows back",
        extra: [[6.6, "Quadcopter: lidar and thermal on the pole hardware", (m) => m.quad.p]],
      },
      POLE: cl(...POLE),
      LDIR,
      LN,
      poles,
      obstacles: [[...cl(...POLE), 0.2]],
      aria: "brush under a power line pulled and chipped, low native grasses planted, and the pole inspected by quadcopter",
    },
    agri: {
      id: "agri",
      chip: "Agriculture",
      title: "Agriculture · vineyard floor",
      problem:
        "Weeds under the vines take water and nutrients. Keeping the strip clean means herbicide, or tillage, or slow hand work, and labor is short at the busy times.",
      solve:
        "The scan maps every row. Hexapods pull the weeds under the vines into a bin and seed a cover crop in the alleys. The quadcopter scans vine health row by row.",
      outcome:
        "A clean under-vine strip with no herbicide, cover crop in the alleys, and a per-vine health map.",
      zone: [rect(-560, C[1] - 130.8, C[0] + 3.4, C[1] + 78)],
      zoneLabel: `Vineyard block, ${(((C[0] + 3.4 + 560) * 208.8) / 1e4).toFixed(1)} ha`,
      zoneAt: [-400, 120],
      flagLabel: "Weeds flagged under the vines",
      flagged: "under-vine weeds flagged, row by row",
      readout: "under-vine weeds → cover crop",
      targets: scatter(
        11,
        8,
        1.4,
        [-5.2, -4.0, 2.8, 4.0],
        (p) => ROWS.some((y) => Math.abs(p[1] - y) < 0.14) && VINE_X(p[0]),
        "weed",
        [0.42, 0.52],
      ).map((t) => t),
      dest: "bin",
      plant: { item: "seed", spots: grid([-4.4, -2.4, -0.4, 1.6], [-2.4, 0, 2.4]), planters: 2 },
      layout: agriLay,
      view: { c: [1.7, -0.3, 0.9], w: 16, az: 18, el: 40 },
      quad: (t) => {
        const u = (Math.sin(t * 0.35) + 1) / 2;
        return cl(-5 + 8.4 * u, -1.2 + 0.0 * u).concat(3.4);
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the vines`,
        problem: "Weeds under the vines take water and nutrients",
        pull: "Pulled by gripper: no herbicide",
        feed: "Into the bin and off the block",
        plant: "Cover crop seeded in the alleys",
        extra: [[7.2, "Quadcopter scans vine health, row by row", (m) => m.quad.p]],
      },
      ROWS,
      VINE_X,
      obstacles: ROWS.flatMap((ry) => {
        const o = [];
        let n = 0;
        for (let x = -29.1; x <= 3.4; x += 1.8) {
          if (x >= -18 && n++ % 4 === 0) o.push([...cl(x + 0.9, ry), 0.06]);
          if (x > -9) o.push([...cl(x, ry), 0.08]);
        }
        o.push([...cl(3.65, ry), 0.08]);
        return o;
      }),
      aria: "weeds under the vine rows pulled into a bin, cover crop seeded in the alleys, and vine health scanned by quadcopter",
    },
    fuels: {
      id: "fuels",
      chip: "Wildfire fuels",
      title: "Wildfire fuels · defensible space",
      problem:
        "Shrubs under trees are ladder fuels: they carry a ground fire up into the canopy and toward the house. Clearing them is hot, slow work, and it has to be done again as they grow back.",
      solve:
        "The scan maps each home's defensible space. Hexapods pull the ladder fuels under the oaks and chip them on site, away from the house.",
      outcome:
        "Defensible space around every structure, cleared and logged, ready for the inspector and the insurer.",
      zone: houses.map((h, i) => blob(h[0], h[1], 30.5, 30.5, 0.0, i, 40)),
      zoneLabel: "100 ft defensible space, each home",
      zoneAt: houses[1],
      flagLabel: "Ladder fuels flagged near homes",
      flagged: "ladder fuels flagged at 6 homes",
      readout: "ladder fuels → defensible space",
      targets: scatter(
        5,
        8,
        1.35,
        [-3.6, -2.4, 2.4, 3.0],
        (p) => d2(p, OAK) < 2.6 && d2(p, OAK) > 0.75 && away(p, keepOut(DEF), 2.0),
        "brush",
        [0.7, 0.82],
      ),
      dest: "chipper",
      plant: { item: null, spots: [], planters: 0 },
      layout: { ...DEF, pile: [5.6, 1.8] },
      view: { c: [-1.6, 2.0, 2.4], w: 20 },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the house`,
        problem: "Ladder fuels: shrubs that carry fire into the oak",
        pull: "Pulled out, roots and all",
        feed: "Chipped on site, away from the house",
        extra: [
          [
            (O) => O.lastWork + 0.6,
            "30 ft from the house: cleared",
            () => {
              const h = cl(...HOUSE.c);
              return [h[0] + 9.1 * 0.67, h[1] - 9.1 * 0.74, 0.2];
            },
          ],
        ],
      },
      HOUSE: { ...HOUSE, c: cl(...HOUSE.c) },
      OAK: cl(...OAK),
      houses,
      obstacles: [
        [...cl(...OAK), 0.32],
        [...cl(7.2, -7.4), 0.3],
        [...cl(-6.0, -8.6), 0.3],
      ],
      aria: "ladder fuels under an oak beside a cabin pulled and chipped on site, leaving 30 ft of defensible space",
    },
    range: {
      id: "range",
      chip: "Rangeland",
      title: "Rangeland · starthistle, fence and water",
      problem:
        "Yellow starthistle takes over pasture. Cattle avoid it once it spines, it is toxic to horses, and each plant left standing seeds the next year. Fences and troughs need checking across miles.",
      solve:
        "The scan maps the infestation. Hexapods pull starthistle before it seeds and bag it, then plug in native bunchgrass. The quadcopter checks troughs and flies the fence line.",
      outcome: "Pasture taken back plant by plant, with water and fence checked on every visit.",
      zone: [blob(-262, 214, 92, 54, 0.08, 2.2)],
      zoneLabel: "Starthistle infestation",
      zoneAt: [-262, 214],
      flagLabel: "Starthistle flagged before it seeds",
      flagged: "starthistle flagged before seed set",
      readout: "starthistle → native bunchgrass",
      targets: scatter(
        3,
        8,
        1.6,
        [-3.6, -1.6, 3.6, 3.4],
        (p) => d2(p, TROUGH) > 2.2 && away(p, keepOut(DEF), 2.0),
        "thistle",
        [0.5, 0.62],
      ),
      dest: "bin",
      plant: { item: "plug", spots: "targets", planters: 2 },
      layout: {
        ...DEF,
        via: [
          [8.5, -2.6],
          [2.4, -1.8],
        ],
      },
      view: { c: [0.6, 0.4, 0.8], w: 14.5 },
      quad: (t) => {
        const tr = cl(TROUGH[0] + 0.4, TROUGH[1] + 0.6);
        if (t < 13) return [tr[0] + Math.sin(t * 0.7) * 0.4, tr[1] + Math.cos(t * 0.7) * 0.3, 2.4];
        const u = Math.min(1, (t - 13) / 9);
        return [
          tr[0] + (C[0] + 7 - tr[0]) * u,
          tr[1] + (C[1] + FENCE_Y - tr[1]) * Math.min(1, u * 3),
          2.4 + 0.6 * Math.min(1, u * 3),
        ];
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the pasture`,
        problem: "Yellow starthistle: cattle avoid it, it's toxic to horses",
        pull: "Pulled before it sets seed",
        feed: "Bagged with the seed heads, hauled off",
        plant: "Native bunchgrass plugged in",
        extra: [
          [6.4, "Quadcopter checks the trough", (m) => m.quad.p],
          [13.6, "…then flies the fence line for breaks", (m) => m.quad.p],
        ],
      },
      FENCE_Y: C[1] + FENCE_Y,
      TROUGH: cl(...TROUGH),
      obstacles: [[...cl(...TROUGH), 1.22]],
      aria: "yellow starthistle pulled before it seeds and bagged, native bunchgrass plugged in, the trough and fence checked by quadcopter",
    },
    forest: {
      id: "forest",
      chip: "Forestry",
      title: "Forestry · post-fire replanting",
      problem:
        "After a high-severity fire, conifers often don't come back on their own: seed sources are gone and resprouting brush shades out what does germinate. Planting crews are scarce and the season is short.",
      solve:
        "The scan maps the burn, snags and survivors. Hexapods pull competing brush and plant conifer seedlings on a grid, and every seedling is mapped.",
      outcome:
        "A replanted stand with a map of every seedling, and survival checks on each return visit.",
      zone: [blob(-318, 150, 150, 112, 0.1, 0.7)],
      zoneLabel: "High-severity burn to replant",
      zoneAt: [-330, 140],
      flagLabel: "Snags, survivors and brush mapped",
      flagged: "burn mapped: snags, survivors, brush",
      readout: "burn scar → conifer seedlings",
      targets: scatter(
        9,
        7,
        1.8,
        [-4.2, -3.6, 3.6, 3.4],
        (p) =>
          away(p, keepOut(DEF), 2.0) &&
          ![-4.5, -1.5, 1.5, 4.5].some((x) => [-4.5, -1.5, 1.5].some((y) => d2(p, [x, y]) < 0.9)),
        "brush",
        [0.58, 0.7],
      ),
      dest: "chipper",
      plant: {
        item: "conifer",
        spots: grid([-4.5, -1.5, 1.5, 4.5], [-4.5, -1.5, 1.5], (p) => away(p, keepOut(DEF), 1.8)),
        planters: 3,
      },
      layout: DEF,
      view: { c: [1.0, 0.6, 1.0], w: 15 },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the burn`,
        problem: "Brush resprouts first and shades out young conifers",
        pull: "Competing brush pulled",
        feed: "Chipped and left as mulch",
        plant: "Conifer seedlings on a 3 m grid",
        extra: [
          [
            (O) => O.lastWork - 1.2,
            "Every seedling mapped for survival checks",
            (m) => {
              const p = m.plants.filter((q) => q.g > 0.9).pop();
              return p ? [...p.p, 0.4] : null;
            },
          ],
        ],
      },
      SNAGS: [
        [-6.2, -5.4, 11],
        [-3.4, -7.0, 9],
        [-8.6, -1.4, 13],
        [7.6, -6.4, 8.5],
        [-10.4, 4.0, 10],
        [2.6, -9.4, 12],
      ].map(([x, y, h]) => [...cl(x, y), h]),
      LOGS: [
        [-5.6, -2.4, 0.45, 5.2],
        [6.8, -4.2, 2.2, 4.0],
        [-2.0, 6.6, 3.0, 3.6],
      ].map(([x, y, a, l]) => [...cl(x, y), a, l]),
      get obstacles() {
        const o = this.SNAGS.map(([x, y]) => [x, y, 0.22]);
        for (const [x, y, ang, l] of this.LOGS)
          for (let f = -0.5; f <= 0.5; f += 0.125)
            o.push([x + Math.cos(ang) * l * f, y + Math.sin(ang) * l * f, 0.22]);
        for (const [x, y] of [
          [-4.6, 6.0],
          [7.4, 0.8],
          [-8.4, -6.2],
        ])
          o.push([...cl(x, y), 0.3]);
        return o;
      },
      aria: "a burned conifer stand: competing brush pulled and chipped, conifer seedlings planted on a grid and each one mapped",
    },
    reclaim: {
      id: "reclaim",
      chip: "Reclamation",
      title: "Reclamation · old quarry",
      problem:
        "A worked-out quarry is bare ground. Rain cuts rills in the slopes, sediment leaves the site, and pampas grass moves in. Permits require erosion control, native cover and monitoring.",
      solve:
        "The scan maps the slopes and drainage. Hexapods pull the pampas grass, stake straw wattles along the contour and sow native seed. The quadcopter resurveys the slope after each storm.",
      outcome: "Slopes held by wattles and native cover, with a survey record for the permit.",
      zone: [rect(C[0] - 130, C[1] - 6.8, C[0] + 70, C[1] + 44)],
      zoneLabel: "Quarry floor and benches",
      zoneAt: [C[0] - 30, C[1] + 18],
      flagLabel: "Pampas grass flagged, slopes surveyed",
      flagged: "pampas flagged, rills mapped",
      readout: "bare quarry → wattles + native seed",
      targets: scatter(
        13,
        7,
        1.8,
        [-4.4, -3.0, 3.6, 3.0],
        (p) =>
          away(p, keepOut(DEF), 2.0) && [-4.3, -1.7, 0.9].every((y) => Math.abs(p[1] - y) > 0.6),
        "pampas",
        [0.72, 0.86],
      ),
      dest: "bin",
      plant: {
        item: "wattle",
        spots: grid([-4.2, -2.0, 0.2, 2.4], [-4.3, -1.7, 0.9], (p) => away(p, keepOut(DEF), 1.6)),
        planters: 3,
      },
      layout: DEF,
      view: { c: [0.0, -1.2, 2.6], w: 18 },
      quad: (t) => {
        const a = t * 0.25;
        return [C[0] - 2 + Math.cos(a) * 3.5, C[1] - 8.5 + Math.sin(a) * 1.2, 11.2];
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the quarry floor`,
        problem: "Pampas grass: its plumes seed the whole site",
        pull: "Pulled before the plumes open",
        feed: "Bagged and hauled off",
        plant: "Straw wattles staked along the contour",
        extra: [
          [7.0, "Quadcopter resurveys the slope for rills", (m) => m.quad.p],
          [
            (O) => O.lastWork - 1.0,
            "Native seed sown behind the wattles",
            (m) => {
              const p = m.plants.filter((q) => q.g > 0.9).pop();
              return p ? [...p.p, 0.2] : null;
            },
          ],
        ],
      },
      BENCHES: [0, 1, 2].map(BENCH),
      aria: "an old quarry: pampas grass pulled and bagged, straw wattles staked along the contour and native seed sown",
    },
    solar: {
      id: "solar",
      chip: "Solar",
      title: "Solar · vegetation and panel checks",
      problem:
        "Weeds grow up between and under the panel rows. Tall growth shades panels and is a fire risk, mowers can't get under the tables, and every panel needs checking for hot spots.",
      solve:
        "The scan maps the rows. Hexapods pull the weeds into a bin and seed a low pollinator mix in the aisles. The quadcopter runs a thermal pass down each row.",
      outcome: "Aisles kept low without herbicide or mowers, and a thermal map of every panel.",
      zone: [rect(C[0] + PV_X0, C[1] - 13, C[0] + PV_X1 + 2, C[1] + 13)],
      zoneLabel: "Panel rows",
      zoneAt: cl(-12, 0),
      flagLabel: "Weeds flagged in the aisles",
      flagged: "weeds flagged in every aisle",
      readout: "weeds → pollinator mix · panels checked",
      targets: scatter(17, 8, 1.4, [-5.2, -1.7, 2.6, 1.7], () => true, "weed", [0.44, 0.54]),
      dest: "bin",
      plant: { item: "seed", spots: grid([-4.4, -2.4, -0.4, 1.6], [-1.0, 1.0]), planters: 2 },
      layout: agriLay,
      view: { c: [0.4, 0, 1.0], w: 17, az: 30, el: 36 },
      quad: (t) => {
        const u = (Math.sin(t * 0.3) + 1) / 2;
        return [...cl(-9 + 11.5 * u, -3.6), 4.6];
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the panel rows`,
        problem: "Weeds in the aisles shade panels and carry fire",
        pull: "Pulled by gripper: no mower, no herbicide",
        feed: "Into the bin and off the site",
        plant: "Low pollinator mix seeded in the aisles",
        extra: [[6.8, "Quadcopter: thermal pass down the row", (m) => m.quad.p]],
      },
      PV: PV.map((y) => C[1] + y),
      PV_X0: C[0] + PV_X0,
      PV_X1: C[0] + PV_X1,
      obstacles: pvPosts,
      aria: "weeds pulled from the aisles of a solar farm into a bin, a pollinator mix seeded, and the panels scanned by quadcopter",
    },
    pipeline: {
      id: "pipeline",
      chip: "Pipeline",
      title: "Pipeline · right-of-way",
      problem:
        "Brush and saplings grow back over buried pipelines. Woody roots threaten the coating, and patrols need a clear corridor and visible markers to spot leaks and digging.",
      solve:
        "The scan maps the corridor. Hexapods cut and pull the woody growth and chip it on site. The quadcopter flies the line with a methane sensor and checks every marker.",
      outcome: "A clear right-of-way, a leak survey of every metre, and a record of each marker.",
      zone: [rect(C[0] - 200, C[1] + PIPE_Y - 7.5, C[0] + 200, C[1] + PIPE_Y + 7.5)],
      zoneLabel: "Right-of-way, 15 m",
      zoneAt: cl(-40, PIPE_Y),
      flagLabel: "Woody growth flagged over the line",
      flagged: "woody growth flagged along the line",
      readout: "brush cleared · line surveyed",
      targets: scatter(
        19,
        8,
        1.6,
        [-4.4, -6.0, 3.6, 3.4],
        (p) => away(p, keepOut(DEF), 2.0) && !(p[0] < -7.6 && p[1] < 0),
        "brush",
        [0.64, 0.8],
      ),
      dest: "chipper",
      plant: { item: null, spots: [], planters: 0 },
      layout: DEF,
      view: { c: [0.6, 0.4, 1.0], w: 17 },
      quad: (t) => {
        const u = (Math.sin(t * 0.22) + 1) / 2;
        return [...cl(-13 + 22 * u, PIPE_Y), 5.2];
      },
      op: {
        arrive: (n) => `A carrier brings ${n} hexapods to the right-of-way`,
        problem: "Woody growth over the pipe: roots and blind spots",
        pull: "Cut and pulled, roots and all",
        feed: "Chipped on site",
        extra: [
          [6.0, "Quadcopter flies the line: methane sensor on", (m) => m.quad.p],
          [
            (O) => O.lastWork + 0.4,
            "Right-of-way clear, markers checked",
            () => [...cl(-6, PIPE_Y), 1.4],
          ],
        ],
      },
      PIPE_Y: C[1] + PIPE_Y,
      VALVE: { x0: C[0] + VALVE.x0, x1: C[0] + VALVE.x1, y0: C[1] + VALVE.y0, y1: C[1] + VALVE.y1 },
      obstacles: [
        [...cl(-6, PIPE_Y), 0.12],
        [...cl(9, PIPE_Y), 0.12],
        ...[
          [-13.5, -6.2],
          [-8.5, -6.2],
          [-8.5, -0.8],
          [-13.5, -0.8],
          [-11, -6.2],
          [-11, -0.8],
          [-8.5, -3.5],
        ].map(([x, y]) => [...cl(x, y), 0.4]),
      ],
      aria: "woody growth cleared from a pipeline right-of-way and chipped, with the line flown by a quadcopter carrying a methane sensor",
    },
  };
  return { ORDER: ["nature", "infra", "agri", "fuels", "range", "forest", "reclaim"], S };
}

// The static world: plinth, 1.6 × 1 km terrain with contour terraces, creek, road, trees, rocks, broom patch,
// hub pad, work cell, plan overlay, laptop. Built once from world.json (exported by the SVG build).

function buildWorld(E, Wd) {
  const { LAND, BOTTOM, PLINTH } = Wd;
  const S = makeKit(E, { n: 4000 });
  const LN = makeLines(E);
  S.begin();
  // plinth: a display base under the land block, with feet
  const pz0 = BOTTOM - 22;
  S.obox(
    [PLINTH.x + PLINTH.w / 2, PLINTH.y + PLINTH.d / 2, pz0 + 11],
    [PLINTH.w / 2, 0, 0],
    [0, PLINTH.d / 2, 0],
    [0, 0, 11],
  );
  for (const [x, y] of [
    [PLINTH.x + 40, PLINTH.y + 40],
    [PLINTH.x + PLINTH.w - 40, PLINTH.y + 40],
    [PLINTH.x + 40, PLINTH.y + PLINTH.d - 40],
    [PLINTH.x + PLINTH.w - 40, PLINTH.y + PLINTH.d - 40],
  ])
    S.disc(x, y, 22, pz0 - 6, 6);
  // land block: sides to BOTTOM, top at 0 (terrain material so the scan coverage shows on it)
  const block = new THREE.Mesh(new THREE.BoxGeometry(LAND.w, LAND.d, -BOTTOM), E.mats.terrain);
  block.position.set(LAND.x + LAND.w / 2, LAND.y + LAND.d / 2, BOTTOM / 2);
  E.scene.add(block);
  for (const z of [-6, -22, -40])
    LN.poly(
      [
        [LAND.x, LAND.y, z],
        [LAND.x + LAND.w, LAND.y, z],
        [LAND.x + LAND.w, LAND.y + LAND.d, z],
        [LAND.x, LAND.y + LAND.d, z],
      ].map(([x, y, zz]) => [x + (x < 0 ? -0.05 : 0.05), y + (y < 0 ? -0.05 : 0.05), zz]),
      "faint",
    );
  // terraces: 10 m contour slabs
  const geos = [];
  for (const hill of Wd.HILLS)
    for (const l of hill.levels) {
      const sh = new THREE.Shape(l.ring.map(([x, y]) => new THREE.Vector2(x, y)));
      const g = new THREE.ExtrudeGeometry(sh, {
        depth: l.h,
        bevelEnabled: false,
        curveSegments: 1,
      });
      g.translate(0, 0, l.z);
      geos.push(g);
    }
  const terr = new THREE.Mesh(mergeGeos(geos), E.mats.terrain);
  E.scene.add(terr);
  // creek and road: flat fills just above the ground
  const flat = (ring, z, mat) => {
    const g = new THREE.ShapeGeometry(
      new THREE.Shape(ring.map(([x, y]) => new THREE.Vector2(x, y))),
    );
    g.translate(0, 0, z);
    const m = new THREE.Mesh(g, mat);
    m.layers.set(1);
    E.scene.add(m);
    return m;
  };
  flat([...Wd.bankL, ...[...Wd.bankR].reverse()], 0.06, E.mats.water);
  flat(Wd.road, 0.04, E.mats.road);
  LN.poly(
    Wd.bankL.map(([x, y]) => [x, y, 0.08]),
    "mid",
    false,
  );
  LN.poly(
    Wd.bankR.map(([x, y]) => [x, y, 0.08]),
    "mid",
    false,
  );
  // trees, rocks
  const ground = makeGround(Wd);
  for (const t of Wd.TREES) {
    const o = [t.x, t.y, ground(t.x, t.y)];
    (t.kind === "oak" ? oak : t.kind === "fir" ? fir : willow)(S, o, t.s);
  }
  for (const [x, y, r] of Wd.ROCKS) rock(S, [x, y, ground(x, y)], r);
  // far textures: grass tufts, broom stipple, planted rows
  for (const [x, y] of Wd.RAW.tufts) {
    LN.seg([x, y, 0.05], [x + 1.6, y + 0.5, 0.05], "faint");
    LN.seg([x, y, 0.05], [x + 0.5, y + 1.6, 0.05], "faint");
  }
  // hub pad with gravel, tufts around it
  const [hx, hy] = Wd.HUB0;
  S.obox([hx + 0.5, hy, 0.03], [6.5, 0, 0], [0, 4.5, 0], [0, 0, 0.03], 0, "rbox");
  let seed = 17;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const grav = [];
  for (let i = 0; i < 400; i++)
    grav.push(hx - 6 + 0.2 + rnd() * 12.6, hy - 4.5 + 0.2 + rnd() * 8.6, 0.065);
  makeDots(E, grav, "lo");
  for (let n = 0; n < 900; n++) {
    const x = hx - 20 + rnd() * 40,
      y = hy - 16 + rnd() * 32;
    if (x > hx - 6.4 && x < hx + 7.4 && y > hy - 4.9 && y < hy + 4.9) continue;
    LN.seg([x, y, 0.004], [x + 0.07, y + 0.03, 0.004], "faint");
    LN.seg([x, y, 0.004], [x + 0.03, y + 0.08, 0.004], "faint");
  }
  person(S, [...Wd.PERSON, 0], Math.PI * 0.85);
  // work cell: tufts, static broom to the west, planted oaks to the east
  const [cx, cy] = Wd.CELL;
  seed = 41;
  for (let n = 0; n < 700; n++) {
    const x = cx + (rnd() * 2 - 1) * 11,
      y = cy + (rnd() * 2 - 1) * 11;
    LN.seg([x, y, 0.004], [x + 0.07, y + 0.03, 0.004], "faint");
    LN.seg([x, y, 0.004], [x + 0.03, y + 0.08, 0.004], "faint");
  }
  // laptop on its own stand beside the diorama (not to scale)
  const L = Wd.LAP,
    ST = Wd.STAND_LAP;
  S.obox(
    [ST.x + ST.w / 2, ST.y + ST.d / 2, BOTTOM - 11],
    [ST.w / 2, 0, 0],
    [0, ST.d / 2, 0],
    [0, 0, 11],
    0,
    "rbox",
  );
  S.obox(
    [L.x + L.w / 2, L.y + L.d / 2, L.z + L.t / 2],
    [L.w / 2, 0, 0],
    [0, L.d / 2, 0],
    [0, 0, L.t / 2],
    0,
    "rbox",
  );
  const kz = L.z + L.t + 0.3;
  for (let r = 0; r < 5; r++)
    for (let c = 0; c < 13; c++) {
      const x = L.hingeX + 30 + r * 23,
        y = L.y + 30 + c * 27.5;
      LN.poly([
        [x, y, kz],
        [x + 19, y, kz],
        [x + 19, y + 26, kz],
        [x, y + 26, kz],
      ]);
    }
  {
    const x = L.hingeX + 30 + 5 * 23 + 10,
      y = L.y + L.d / 2 - 60;
    LN.poly([
      [x, y, kz],
      [x + 70, y, kz],
      [x + 70, y + 120, kz],
      [x, y + 120, kz],
    ]);
  }
  const ev = [-Math.sin(L.lean), 0, Math.cos(L.lean)],
    nv = [Math.cos(L.lean), 0, Math.sin(L.lean)];
  const hinge = (y) => [L.hingeX, y, L.z + L.t];
  const lidC = [
    hinge(0)[0] + (ev[0] * L.lidH) / 2 - nv[0] * 4,
    L.y + L.d / 2,
    hinge(0)[2] + (ev[2] * L.lidH) / 2 - nv[2] * 4,
  ];
  S.obox(
    lidC,
    [0, 0, 0]
      .map((_, i) => [0, 0, 0][i])
      .map(() => 0)
      .map((v, i) => (i === 1 ? 0 : 0)) && [0, L.d / 2, 0],
    ev.map((v) => (v * L.lidH) / 2),
    nv.map((v) => v * 4),
  );
  S.rod([L.hingeX, L.y + 30, L.z + L.t], [L.hingeX, L.y + L.d - 30, L.z + L.t], 6);
  // the screen: a plane carrying the live twin texture
  const sw = L.d - 2 * L.border,
    shh = L.lidH - 2 * L.border;
  const screenGeo = new THREE.PlaneGeometry(sw, shh);
  const screenMat = new THREE.MeshBasicMaterial({ color: 0xffffff, map: null, toneMapped: false });
  const screen = new THREE.Mesh(screenGeo, screenMat);
  const sc = [
    hinge(0)[0] + ev[0] * (L.lidH / 2) + nv[0] * 0.6,
    L.y + L.d / 2,
    hinge(0)[2] + ev[2] * (L.lidH / 2) + nv[2] * 0.6,
  ];
  screen.position.set(...sc);
  // orient in render space (y mirrored): width along +y, up the lid, facing nv; a local y-flip undoes the scene's mirror
  const B3 = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(...ev),
    new THREE.Vector3(...nv),
  );
  const Sf = new THREE.Matrix4().makeScale(1, -1, 1);
  const Rl = new THREE.Matrix4().multiplyMatrices(Sf, B3).multiply(Sf);
  screen.quaternion.setFromRotationMatrix(Rl);
  screen.scale.set(1, -1, 1);
  E.scene.add(screen);
  S.end();
  LN.build();
  return { ground, screen, screenMat, screenSize: [sw, shh], screenCenter: sc };
}

function makeGround(Wd) {
  const levels = Wd.HILLS.flatMap((h) => h.levels);
  const inside = ([x, y], ring) => {
    let c = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i],
        [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  const f = (x, y) => {
    let z = 0;
    for (const l of levels) if (l.z + l.h > z && inside([x, y], l.ring)) z = l.z + l.h;
    return z;
  };
  f.inside = inside;
  return f;
}

function mergeGeos(geos) {
  // non-indexed concat of positions (normals are derived in the shader)
  let n = 0;
  const parts = geos.map((g) => {
    const ng = g.index ? g.toNonIndexed() : g;
    n += ng.attributes.position.count;
    return ng;
  });
  const pos = new Float32Array(n * 3);
  let o = 0;
  for (const g of parts) {
    pos.set(g.attributes.position.array, o);
    o += g.attributes.position.array.length;
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return out;
}

// Per-scenario static scenery: parcel-scale features (stipple, lines, poles, houses, benches) and the
// cell-scale props around the work cell. One group per scenario; only the active one is visible.

function buildScenes(E, Wd, SCN, ground, opts = {}) {
  const C = Wd.CELL,
    LAND = Wd.LAND;
  const cl = (x, y) => [C[0] + x, C[1] + y];
  const inLand = ([x, y], m = 6) =>
    x > LAND.x + m && x < LAND.x + LAND.w - m && y > LAND.y + m && y < LAND.y + LAND.d - m;
  const nearCell = ([x, y], m = 10) => Math.abs(x - C[0]) < m && Math.abs(y - C[1]) < m;
  const groups = {};
  let seed = 1;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  // opts.only: build just these scenarios; opts.near {c, r}: keep only what lies within r of c (a compact vignette)
  const want = (id) => !opts.only || opts.only.includes(id);
  const NEAR = opts.near || null;
  let nearR = NEAR ? NEAR.r : Infinity;
  const ok2 = (x, y) => !NEAR || Math.hypot(x - NEAR.c[0], y - NEAR.c[1]) <= nearR;
  function open(id, s) {
    seed = s;
    const g = new THREE.Group();
    g.visible = false;
    E.scene.add(g);
    groups[id] = g;
    const k0 = makeKit(E, { n: NEAR ? 1600 : 3200, parent: g });
    const ln0 = makeLines(E, { parent: g });
    k0.begin();
    const dots = (pts, tone, z) => {
      const q = pts.filter(([x, y]) => ok2(x, y));
      if (q.length)
        makeDots(
          E,
          q.flatMap(([x, y, zz]) => [x, y, zz ?? z]),
          tone,
          g,
        );
    };
    const mesh = (geo, mat, layer = 0) => {
      if (NEAR) {
        geo.computeBoundingBox();
        const b = geo.boundingBox;
        if (
          !ok2((b.min.x + b.max.x) / 2, (b.min.y + b.max.y) / 2) &&
          !ok2(
            Math.max(b.min.x, Math.min(NEAR.c[0], b.max.x)),
            Math.max(b.min.y, Math.min(NEAR.c[1], b.max.y)),
          )
        )
          return null;
      }
      const m = new THREE.Mesh(geo, mat);
      if (layer) m.layers.set(layer);
      g.add(m);
      return m;
    };
    const k = !NEAR
      ? k0
      : {
          ...k0,
          obox: (c, ...r) => ok2(c[0], c[1]) && k0.obox(c, ...r),
          link: (A, B, ...r) => ok2((A[0] + B[0]) / 2, (A[1] + B[1]) / 2) && k0.link(A, B, ...r),
          rod: (A, B, ...r) => ok2(A[0], A[1]) && k0.rod(A, B, ...r),
          sph: (c, ...r) => ok2(c[0], c[1]) && k0.sph(c, ...r),
          cyl: (x, y, ...r) => ok2(x, y) && k0.cyl(x, y, ...r),
          cone: (x, y, ...r) => ok2(x, y) && k0.cone(x, y, ...r),
          disc: (x, y, ...r) => ok2(x, y) && k0.disc(x, y, ...r),
        };
    const ln = !NEAR
      ? ln0
      : {
          ...ln0,
          seg: (a, b, tone) => ok2((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) && ln0.seg(a, b, tone),
          poly(pts, tone, closed = true) {
            for (let i = 0; i < pts.length - (closed ? 0 : 1); i++)
              this.seg(pts[i], pts[(i + 1) % pts.length], tone);
          },
        };
    return {
      g,
      k,
      ln,
      dots,
      mesh,
      close: () => {
        k0.end();
        ln0.build();
      },
    };
  }
  const inRing = ([x, y], ring) => ground.inside([x, y], ring);
  const fill = (ring, n, ok = () => true) => {
    const xs = ring.map((p) => p[0]),
      ys = ring.map((p) => p[1]);
    const x0 = Math.min(...xs),
      x1 = Math.max(...xs),
      y0 = Math.min(...ys),
      y1 = Math.max(...ys);
    const out = [];
    for (let i = 0; i < n * 8 && out.length < n; i++) {
      const p = [x0 + rnd() * (x1 - x0), y0 + rnd() * (y1 - y0)];
      if (inRing(p, ring) && inLand(p) && ok(p)) out.push(p);
    }
    return out;
  };
  // house: a gabled box (extruded profile) with overhanging roof slabs, door and windows
  function house(s, c, w, d, h, yaw = 0) {
    const r = 1.5;
    const sh = new THREE.Shape(
      [
        [-d / 2, 0],
        [d / 2, 0],
        [d / 2, h],
        [0, h + r],
        [-d / 2, h],
      ].map(([u, v]) => new THREE.Vector2(u, v)),
    );
    const geo = new THREE.ExtrudeGeometry(sh, { depth: w, bevelEnabled: false });
    geo.applyMatrix4(
      new THREE.Matrix4().makeBasis(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(1, 0, 0),
      ),
    );
    geo.translate(-w / 2, 0, 0);
    geo.rotateZ(yaw);
    const z0 = ground(c[0], c[1]);
    geo.translate(c[0], c[1], z0);
    s.mesh(geo, E.mats.terrain);
    const R = (v) => [
      v[0] * Math.cos(yaw) - v[1] * Math.sin(yaw),
      v[0] * Math.sin(yaw) + v[1] * Math.cos(yaw),
      v[2],
    ];
    const P = (v) => {
      const q = R(v);
      return [c[0] + q[0], c[1] + q[1], z0 + q[2]];
    };
    const slope = r / (d / 2),
      ov = 0.4,
      ze = h - ov * slope;
    for (const sg of [-1, 1]) {
      const ey = [0, (sg * (d / 2 + ov)) / 2, (ze - (h + r)) / 2];
      const len = Math.hypot(ey[1], ey[2]);
      const nrm = [0, ((sg * -ey[2]) / len) * sg * sg, Math.abs(ey[1]) / len];
      const nz = [0, (sg * Math.abs(ey[2])) / len, Math.abs(ey[1]) / len];
      const cc = [0, ey[1], h + r + ey[2]];
      s.k.obox(
        P([cc[0] + nz[0] * 0.06, cc[1] + nz[1] * 0.06, cc[2] + nz[2] * 0.06]),
        R([w / 2 + 0.3, 0, 0]),
        R(ey),
        R(nz.map((v) => v * 0.05)),
      );
      void nrm;
    }
    s.k.obox(
      P([w / 2 + 0.004, -0.8, 1.05]),
      R([0.006, 0, 0]),
      R([0, 0.48, 0]),
      [0, 0, 1.02],
      0,
      "dark",
    );
    for (const y of [1.4])
      s.k.obox(
        P([w / 2 + 0.004, y, 1.55]),
        R([0.006, 0, 0]),
        R([0, 0.6, 0]),
        [0, 0, 0.5],
        0,
        "dark",
      );
    for (const x of [-w / 4, w / 4])
      s.k.obox(
        P([x, d / 2 + 0.004, 1.55]),
        R([0.6, 0, 0]),
        R([0, 0.006, 0]),
        [0, 0, 0.5],
        0,
        "dark",
      );
    s.k.obox(P([-w / 3, 0.9, h + 1.1]), R([0.3, 0, 0]), R([0, 0.3, 0]), [0, 0, 0.9]);
  }
  function cow(k, o, h) {
    const R = (v) => [
      o[0] + v[0] * Math.cos(h) - v[1] * Math.sin(h),
      o[1] + v[0] * Math.sin(h) + v[1] * Math.cos(h),
      o[2] + v[2],
    ];
    const V = (v) => [
      v[0] * Math.cos(h) - v[1] * Math.sin(h),
      v[0] * Math.sin(h) + v[1] * Math.cos(h),
      v[2],
    ];
    k.obox(R([0, 0, 1.05]), V([0.95, 0, 0]), V([0, 0.3, 0]), [0, 0, 0.3], 0, "rbox");
    for (const x of [-0.7, 0.7])
      for (const y of [-0.18, 0.18]) {
        const p = R([x, y, 0]);
        k.cyl(p[0], p[1], 0.06, o[2], 0.8);
      }
    k.link(R([0.95, 0, 1.15]), R([1.35, 0, 0.55]), 0.22);
    k.obox(R([1.42, 0, 0.42]), V([0.2, 0, 0]), V([0, 0.12, 0]), [0, 0, 0.12], 0, "rbox");
    k.link(R([-0.95, 0, 1.2]), R([-1.05, 0, 0.6]), 0.03);
  }

  // ── nature: broom stipple, planted rows, the static broom and saplings around the cell ──
  if (want("nature")) {
    const s = open("nature", 41);
    s.dots(Wd.RAW.broom, "mid", 0.6);
    s.dots(Wd.RAW.planted, "lo", 0.3);
    for (const c of Wd.CELL_STATIC) {
      if (c.kind === "broom") shrub(s.k, [c.x, c.y, 0], { seed: c.seed, scale: 0.95 });
      else sapling(s.k, [c.x, c.y, 0]);
    }
    s.close();
  }

  // ── infrastructure: poles every 60 m, three conductors with sag, brush stipple in the corridor ──
  if (want("infra")) {
    const s = open("infra", 53),
      I = SCN.infra;
    if (NEAR) nearR = NEAR.poleR || NEAR.r;
    const tips = I.poles.map((p) => POLE_TIPS([p[0], p[1], ground(p[0], p[1])], I.LN));
    I.poles.forEach((p) => pole(s.k, [p[0], p[1], ground(p[0], p[1])], I.LN));
    for (let i = 0; i < tips.length - 1; i++)
      for (let j = 0; j < 3; j++) {
        const a = tips[i][j],
          b = tips[i + 1][j];
        const pts = [];
        for (let q = 0; q <= 12; q++) {
          const u = q / 12;
          pts.push([
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u - 0.9 * 4 * u * (1 - u),
          ]);
        }
        s.ln.poly(pts, "mid", false);
      }
    const brush = [];
    for (let i = 0; i < 4000 && brush.length < 1500; i++) {
      const a = I.poles[0],
        b = I.poles[I.poles.length - 1],
        u = rnd(),
        w = (rnd() * 2 - 1) * 13;
      const p = [a[0] + (b[0] - a[0]) * u + I.LN[0] * w, a[1] + (b[1] - a[1]) * u + I.LN[1] * w];
      if (!inLand(p) || nearCell(p, 12) || rnd() > 0.55 + 0.45 * Math.sin(u * 19)) continue;
      brush.push([p[0], p[1], ground(p[0], p[1]) + 0.6]);
    }
    s.dots(brush, "mid");
    // more brush up the line beyond the cell (the next cells)
    for (let i = 0; i < 16; i++) {
      const along = 7 + rnd() * 12,
        w = (rnd() * 2 - 1) * 2.4;
      shrub(
        s.k,
        [
          C[0] + 0.6 + I.LDIR[0] * along + I.LN[0] * w,
          C[1] + 1.2 + I.LDIR[1] * along + I.LN[1] * w,
          0,
        ],
        { seed: rnd() * 9, scale: 0.7 + rnd() * 0.2 },
      );
    }
    s.close();
    nearR = NEAR ? NEAR.r : Infinity;
  }

  // ── agriculture: vine rows (every other row at parcel scale), trellis and canopy near the cell ──
  if (want("agri")) {
    const s = open("agri", 61),
      A = SCN.agri,
      z = SCN.agri.zone[0];
    const y0 = Math.min(...z.map((p) => p[1])),
      y1 = Math.max(...z.map((p) => p[1])),
      x0 = Math.min(...z.map((p) => p[0])),
      x1 = Math.max(...z.map((p) => p[0]));
    for (
      let y = C[1] - 3.6 - 2.4 * Math.floor((C[1] - 3.6 - y0) / 2.4), n = 0;
      y < y1;
      y += 2.4, n++
    ) {
      if (Math.abs(y - C[1]) < 7.5) {
        s.ln.seg([x0 + 2, y, 1.3], [C[0] - 78, y, 1.3], "lo");
        continue;
      }
      s.ln.seg([x0 + 2, y, 1.3], [x1, y, 1.3], "lo");
    }
    for (const ry of A.ROWS) {
      const y = C[1] + ry,
        xa = -18,
        xb = 3.4;
      for (const zz of [0.95, 1.4, 1.8]) s.ln.seg([C[0] + xa, y, zz], [C[0] + xb, y, zz], "lo");
      s.ln.seg([C[0] + xa - 60, y, 1.3], [C[0] + xa, y, 1.3], "lo");
      let n = 0;
      for (let x = -29.1; x <= xb; x += 1.8) {
        if (x < xa) continue;
        const p = [C[0] + x, y];
        s.k.cyl(p[0], p[1], 0.035, 0, 0.95);
        s.k.link([p[0] - 0.5, y, 0.95], [p[0] + 0.5, y, 0.95], 0.035);
        s.k.obox(
          [p[0], y + Math.sin(x * 3.1 + ry) * 0.03, 1.46],
          [0.86, 0, 0],
          [0, 0.17 + 0.02 * Math.sin(x * 2.3), 0],
          [0, 0, 0.36 + 0.04 * Math.sin(x * 1.7 + ry)],
          0,
          "rbox",
        );
        if (n++ % 4 === 0) s.k.cyl(p[0] + 0.9, y, 0.04, 0, 1.95);
      }
      s.k.cyl(C[0] + xb + 0.25, y, 0.06, 0, 2.0);
      s.k.link([C[0] + xb + 0.25, y, 1.9], [C[0] + xb + 1.3, y, 0], 0.012);
    }
    s.close();
  }

  // ── fuels: six homes with 100 ft rings (plan), oaks and ladder-fuel stipple; the cabin, its oak, more oaks ──
  if (want("fuels")) {
    const s = open("fuels", 71),
      F = SCN.fuels;
    F.houses.forEach((h, i) => {
      if (i === 0) house(s, F.HOUSE.c, F.HOUSE.w, F.HOUSE.d, F.HOUSE.h, 0);
      else house(s, h, 8 + rnd() * 3, 6 + rnd() * 2, 2.8, rnd() * 3);
    });
    const lf = [];
    F.houses.forEach((h, i) => {
      for (let n = 0; n < 220; n++) {
        const a = rnd() * 6.283,
          r = 9 + rnd() * 24;
        const p = [h[0] + Math.cos(a) * r, h[1] + Math.sin(a) * r];
        if (inLand(p) && !nearCell(p, 12)) lf.push([p[0], p[1], ground(p[0], p[1]) + 0.5]);
      }
      if (i)
        for (let n = 0; n < 4; n++) {
          const a = rnd() * 6.283,
            r = 10 + rnd() * 16;
          oak(s.k, [h[0] + Math.cos(a) * r, h[1] + Math.sin(a) * r, 0], 0.7 + rnd() * 0.3);
        }
    });
    s.dots(lf, "mid");
    // the oak beside the cabin: limbed high enough to see under
    s.k.cyl(F.OAK[0], F.OAK[1], 0.3, 0, 3.6);
    s.k.link([F.OAK[0], F.OAK[1], 3.0], [F.OAK[0] - 1.2, F.OAK[1] + 0.4, 4.4], 0.18);
    s.k.link([F.OAK[0], F.OAK[1], 3.2], [F.OAK[0] + 1.1, F.OAK[1] - 0.7, 4.6], 0.16);
    s.k.sph([F.OAK[0], F.OAK[1], 5.8], 2.5);
    s.k.sph([F.OAK[0] + 1.4, F.OAK[1] - 0.9, 5.2], 1.7);
    s.k.sph([F.OAK[0] - 1.5, F.OAK[1] + 0.6, 5.0], 1.5);
    // the next oaks over, still with ladder fuels under them
    for (const [x, y, sc] of [
      [7.2, -7.4, 0.7],
      [-6.0, -8.6, 0.66],
    ]) {
      oak(s.k, [...cl(x, y), 0], sc);
      for (let n = 0; n < 6; n++) {
        const a = n * 1.05 + rnd(),
          r = 0.9 + rnd() * 1.4;
        shrub(s.k, [C[0] + x + Math.cos(a) * r, C[1] + y + Math.sin(a) * r, 0], {
          seed: rnd() * 9,
          scale: 0.72 + rnd() * 0.12,
        });
      }
    }
    s.close();
  }

  // ── rangeland: pasture fence, troughs, starthistle stipple, cows ──
  if (want("range")) {
    const s = open("range", 83),
      R = SCN.range;
    const fy = R.FENCE_Y,
      xa = LAND.x + 40,
      xb = 130,
      ys = LAND.y + LAND.d - 40;
    for (const z of [0.55, 0.85, 1.15]) {
      s.ln.seg([xa, fy, z], [xb, fy, z], "mid");
      s.ln.seg([xa, fy, z], [xa, ys, z], "mid");
      s.ln.seg([-60, fy, z], [-60, ys, z], "mid");
    }
    for (let x = C[0] - 24; x <= C[0] + 24; x += 3) s.k.cyl(x, fy, 0.05, 0, 1.25);
    const tr = R.TROUGH;
    for (const [x, y] of [tr, [-150, 330]]) {
      s.k.cyl(x, y, 1.2, 0, 0.55);
      const w = s.mesh(new THREE.CircleGeometry(1.1, 40).translate(x, y, 0.56), E.mats.water, 1);
      void w;
      s.k.obox([x + 1.0, y - 0.6, 0.62], [0.16, 0, 0], [0, 0.12, 0], [0, 0, 0.1]);
      s.k.rod([x + 1.25, y - 0.7, 0], [x + 1.25, y - 0.7, 0.68], 0.03);
    }
    const st = fill(R.zone[0], 1400, (p) => !nearCell(p, 10)).map(([x, y]) => [
      x,
      y,
      ground(x, y) + 0.4,
    ]);
    s.dots(st, "mid");
    for (let n = 0; n < 16; n++) {
      const p = [-15 + rnd() * 30, -4.6 + rnd() * 16];
      if (Math.abs(p[0]) < 8.5 && p[1] < 9) continue;
      shrub(s.k, [...cl(p[0], p[1]), 0], {
        kind: "thistle",
        seed: rnd() * 9,
        scale: 0.5 + rnd() * 0.12,
      });
    }
    cow(s.k, [...cl(-8.6, 4.4), 0], 0.5);
    cow(s.k, [...cl(7.8, -9.2), 0], 2.6);
    for (let n = 0; n < 9; n++) cow(s.k, [-200 + rnd() * 150, 230 + rnd() * 150, 0], rnd() * 6);
    s.close();
  }

  // ── forestry: snags as ticks across the burn, ash; snags and logs in the cell ──
  if (want("forest")) {
    const s = open("forest", 97),
      F = SCN.forest;
    for (const p of fill(F.zone[0], 650, (q) => !nearCell(q, 14))) {
      const z = ground(p[0], p[1]);
      s.ln.seg([p[0], p[1], z], [p[0] + 0.2, p[1], z + 7 + rnd() * 8], "mid");
    }
    s.dots(
      fill(F.zone[0], 1600).map(([x, y]) => [x, y, ground(x, y) + 0.2]),
      "lo",
    );
    for (const [x, y, h] of F.SNAGS) {
      s.k.obox([x, y, h / 2], [0.17, 0, 0], [0, 0.17, 0], [0, 0, h / 2], 0, "dark");
      s.k.obox([x + 0.5, y, h * 0.62], [0.5, 0, 0.18], [0, 0.05, 0], [0, 0, 0.05], 0, "dark");
      s.k.obox([x, y - 0.4, h * 0.45], [0, 0.4, 0.2], [0.05, 0, 0], [0, 0, 0.05], 0, "dark");
    }
    for (const [x, y, a, l] of F.LOGS)
      s.k.obox(
        [x, y, 0.2],
        [(Math.cos(a) * l) / 2, (Math.sin(a) * l) / 2, 0],
        [-Math.sin(a) * 0.2, Math.cos(a) * 0.2, 0],
        [0, 0, 0.2],
        0,
        "dark",
      );
    for (const [x, y] of [
      [-4.6, 6.0],
      [7.4, 0.8],
      [-8.4, -6.2],
    ])
      s.k.cyl(C[0] + x, C[1] + y, 0.28, 0, 0.4);
    s.close();
  }

  // ── reclamation: quarry benches (extruded), pampas stipple on the floor and the benches ──
  if (want("reclaim")) {
    const s = open("reclaim", 109),
      Q = SCN.reclaim;
    const clipRing = (ring) =>
      NEAR
        ? ring.map(([x, y]) => [
            Math.max(NEAR.c[0] - NEAR.r, Math.min(NEAR.c[0] + NEAR.r, x)),
            Math.max(NEAR.c[1] - NEAR.r, Math.min(NEAR.c[1] + NEAR.r, y)),
          ])
        : ring;
    for (const b of Q.BENCHES) {
      const g = new THREE.ExtrudeGeometry(
        new THREE.Shape(clipRing(b.ring).map(([x, y]) => new THREE.Vector2(x, y))),
        { depth: b.h, bevelEnabled: false },
      );
      g.translate(0, 0, b.z);
      s.mesh(g, E.mats.terrain);
    }
    s.dots(
      fill(Q.zone[0], 700, (p) => !nearCell(p, 10)).map(([x, y]) => [x, y, 0.9]),
      "mid",
    );
    const top = Q.BENCHES[2];
    s.dots(
      fill(top.ring, 500).map(([x, y]) => [x, y, top.z + top.h + 0.6]),
      "mid",
    );
    for (const [x, y, z] of [
      [-9, -8.8, 3.6],
      [5.6, -9.0, 3.6],
      [-3.2, -12.8, 7.2],
      [9.4, -13, 7.2],
      [-12.2, 3.6, 0],
      [10.4, 5.4, 0],
    ])
      shrub(s.k, [...cl(x, y), z], { kind: "pampas", seed: x * 0.7, scale: 0.8 });
    // cut faces: vertical hatching on each bench face
    for (const b of Q.BENCHES) {
      const y = b.ring[2][1] + 0.02;
      for (let x = C[0] - 40; x < C[0] + 40; x += 0.9 + rnd() * 0.8)
        s.ln.seg(
          [x, y, b.z + 0.2 + rnd() * 0.6],
          [x + (rnd() - 0.5) * 0.3, y, b.z + b.h - 0.2 - rnd() * 0.8],
          "faint",
        );
    }
    // rills down the bench faces
    for (let i = 0; i < 9; i++) {
      const x = C[0] - 14 + i * 3.4 + rnd();
      for (const b of Q.BENCHES) {
        const y = b.ring[2][1];
        s.ln.poly(
          [
            [x, y + 0.02, b.z + b.h],
            [x + 0.2, y + 0.02, b.z + b.h * 0.5],
            [x - 0.1, y + 0.02, b.z + 0.05],
          ],
          "lo",
          false,
        );
      }
    }
    s.close();
  }

  // ── solar: fixed-tilt panel tables on posts, module lines, an inverter skid, the perimeter fence ──
  if (want("solar")) {
    const s = open("solar", 127),
      P = SCN.solar;
    if (NEAR) nearR = NEAR.wide || NEAR.r;
    const t = (25 * Math.PI) / 180,
      hw = 1.1,
      dz = Math.sin(t) * hw,
      dy = Math.cos(t) * hw,
      z0 = 0.72;
    const nz = [0, Math.sin(t), Math.cos(t)];
    for (const y of P.PV)
      for (let x = P.PV_X0; x < P.PV_X1 - 0.1; x += 4.3) {
        const L = 4.0,
          cx = x + L / 2,
          cz = z0 + dz;
        s.k.obox(
          [cx, y, cz],
          [L / 2, 0, 0],
          [0, dy, -dz],
          nz.map((v) => v * 0.035),
          0,
          "dark",
        );
        for (const sg of [-1, 1]) {
          const py = y + sg * 0.85,
            h = sg < 0 ? z0 + 2 * dz - 0.12 : z0 - 0.04;
          s.k.cyl(x + 0.25, py, 0.05, 0, h);
          s.k.cyl(x + L - 0.25, py, 0.05, 0, h);
        }
        const off = (p) => [p[0] + nz[0] * 0.05, p[1] + nz[1] * 0.05, p[2] + nz[2] * 0.05];
        for (let u = 1; u < 4; u++)
          s.ln.seg(off([x + u, y - dy, cz + dz]), off([x + u, y + dy, cz - dz]), "faint");
        s.ln.seg(off([x, y, cz]), off([x + L, y, cz]), "faint");
      }
    s.k.obox([C[0] + 7.6, C[1] - 8.4, 0.75], [0.9, 0, 0], [0, 0.55, 0], [0, 0, 0.75], 0, "rbox");
    s.k.obox([C[0] + 7.6, C[1] - 8.4, 0.08], [1.4, 0, 0], [0, 1.0, 0], [0, 0, 0.08]);
    const fy = C[1] - 14.2;
    for (const z of [0.5, 1.1, 1.7]) s.ln.seg([P.PV_X0 - 6, fy, z], [C[0] + 16, fy, z], "mid");
    for (let x = P.PV_X0 - 6; x <= C[0] + 16; x += 3) s.k.cyl(x, fy, 0.035, 0, 1.8);
    // aisles: grass texture between rows away from the cell
    const gr = [];
    for (let i = 0; i < 1600; i++) {
      const y = P.PV[Math.floor(rnd() * 4)] + 3.6 * (rnd() < 0.5 ? -1 : 1) + (rnd() - 0.5) * 3.6,
        x = P.PV_X0 + rnd() * (P.PV_X1 - P.PV_X0);
      if (!nearCell([x, y], 7)) gr.push([x, y, 0.18]);
    }
    s.dots(gr, "lo");
    s.close();
    nearR = NEAR ? NEAR.r : Infinity;
  }

  // ── pipeline: the cleared right-of-way between tree lines, the buried line dashed, markers, a fenced valve site ──
  if (want("pipeline")) {
    const s = open("pipeline", 139),
      P = SCN.pipeline,
      y = P.PIPE_Y;
    if (NEAR) nearR = NEAR.wide || NEAR.r;
    const xa = C[0] - 60,
      xb = C[0] + 60;
    for (let x = xa; x < xb; x += 1.6)
      if (x < P.VALVE.x0 - 0.3 || x > P.VALVE.x1 + 0.3)
        s.ln.seg([x, y, 0.02], [x + 0.9, y, 0.02], "lit");
    for (let x = C[0] - 6 - 30 * 2; x <= xb; x += 15) {
      if (x > P.VALVE.x0 - 1 && x < P.VALVE.x1 + 1) continue;
      s.k.cyl(x, y, 0.05, 0, 1.25);
      s.k.obox([x, y, 1.12], [0.03, 0, 0], [0, 0.16, 0], [0, 0, 0.14], 0, "dark");
    }
    // valve site: risers, a crossover with a hand wheel, a fence
    const V = P.VALVE,
      vx = (V.x0 + V.x1) / 2;
    for (const x of [vx - 1.2, vx + 1.2]) s.k.link([x, y, 0], [x, y, 1.0], 0.16);
    s.k.link([vx - 1.2, y, 1.0], [vx + 1.2, y, 1.0], 0.16);
    s.k.link([vx, y, 1.0], [vx, y, 1.55], 0.05);
    s.k.disc(vx, y, 0.32, 1.58, 0.03);
    s.k.obox([vx - 1.6, V.y0 + 0.9, 0.5], [0.4, 0, 0], [0, 0.3, 0], [0, 0, 0.5], 0, "rbox");
    const ring = [
      [V.x0, V.y0],
      [V.x1, V.y0],
      [V.x1, V.y1],
      [V.x0, V.y1],
    ];
    for (const z of [0.5, 1.1, 1.7])
      s.ln.poly(
        ring.map(([a, b]) => [a, b, z]),
        "mid",
      );
    for (let i = 0; i < 4; i++) {
      const a = ring[i],
        b = ring[(i + 1) % 4],
        L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      for (let d = 0; d < L; d += 1.8)
        s.k.cyl(a[0] + ((b[0] - a[0]) * d) / L, a[1] + ((b[1] - a[1]) * d) / L, 0.035, 0, 1.8);
    }
    // tree lines either side of the 15 m right-of-way, brush regrowth further along the corridor
    for (let x = xa; x < xb; x += 3.4 + rnd() * 2.8)
      for (const sg of [-1, 1]) {
        const yy = y + sg * (12 + rnd() * 5);
        if (Math.hypot(x - C[0], yy - C[1]) < 17) continue;
        (rnd() < 0.6 ? oak : fir)(s.k, [x, yy, 0], 0.24 + rnd() * 0.1);
      }
    for (let i = 0; i < 26; i++) {
      const x = xa + rnd() * (xb - xa),
        yy = y + (rnd() * 2 - 1) * 6.5;
      if (nearCell([x, yy], 9) || (x > V.x0 - 1 && x < V.x1 + 1 && yy > V.y0 - 1 && yy < V.y1 + 1))
        continue;
      shrub(s.k, [x, yy, 0], { kind: "brush", seed: rnd() * 9, scale: 0.6 + rnd() * 0.25 });
    }
    s.close();
    nearR = NEAR ? NEAR.r : Infinity;
  }

  let cur = "";
  return {
    show(id) {
      if (id === cur) return;
      cur = id;
      for (const [k, g] of Object.entries(groups)) g.visible = k === id;
    },
  };
}

// Story: every robot, vehicle and plant at (stage, t). Pure state, no drawing; ported from the SVG version.
function makeStory(Wd, ground, SCN, opts = {}) {
  const HUB0 = Wd.HUB0,
    CELL = Wd.CELL;
  const ORDER = ["plan", "deploy", "scan", "operate"];
  const DUR = { plan: 21, deploy: 26, scan: 22, operate: 30 };
  let camAt = () => null;
  // ── helpers ──
  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const ease = (v) => {
    v = clamp01(v);
    return v * v * (3 - 2 * v);
  };
  const seg = (t, a, b) => ease((t - a) / (b - a));
  const lin = (t, a, b) => clamp01((t - a) / (b - a));
  const lerp = (a, b, u) => a + (b - a) * u;
  const lerp2 = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u)];
  const lerp3 = (a, b, u) => [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u)];
  const dist2 = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  const hdg = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);
  const hubW = ([x, y, z = 0]) => [HUB0[0] + x, HUB0[1] + y, z];
  function along(route, u) {
    const lens = route.slice(1).map((p, i) => dist2(route[i], p));
    const total = lens.reduce((a, b) => a + b, 0);
    let s = clamp01(u) * total;
    for (let i = 0; i < lens.length; i++) {
      if (s <= lens[i] || i === lens.length - 1) {
        const f = lens[i] ? Math.min(1, s / lens[i]) : 0;
        return { p: lerp2(route[i], route[i + 1], f), h: hdg(route[i], route[i + 1]), len: total };
      }
      s -= lens[i];
    }
  }

  // ── scan plan ──
  const VT = Wd.MODEL.vtol,
    QD = Wd.MODEL.quad;
  const PASSES = Math.floor((Wd.LAND.d - 60) / VT.spacingM) + 1;
  const Y0 = Wd.LAND.y + 30;
  const passY = (i) => Y0 + VT.spacingM * i;
  const XA = Wd.LAND.x - 40,
    XB = Wd.LAND.x + Wd.LAND.w + 40;
  const TP0 = 4.0,
    TP1 = 15.4,
    TPD = (TP1 - TP0) / PASSES,
    RUN = 0.86;
  let roofV, roofQ;
  function vtolAt(t) {
    const ALT = VT.altM;
    if (t < 0.8) return { p: roofV, h: 0, spin: t > 0.3 };
    if (t < 3.0) {
      const u = seg(t, 0.8, 3.0);
      return { p: [roofV[0], roofV[1], lerp(roofV[2], ALT, u)], h: 0, spin: true };
    }
    const start = [XA, passY(0)];
    if (t < TP0) {
      const u = seg(t, 3.0, TP0);
      const q = lerp2([roofV[0], roofV[1]], start, u);
      return { p: [...q, ALT], h: hdg([roofV[0], roofV[1]], start), spin: true };
    }
    if (t < TP1) {
      const v = (t - TP0) / TPD,
        i = Math.min(PASSES - 1, Math.floor(v)),
        f = v - i;
      const dir = i % 2 ? -1 : 1,
        xs = dir > 0 ? [XA, XB] : [XB, XA];
      if (f < RUN) {
        const x = lerp(xs[0], xs[1], f / RUN);
        return { p: [x, passY(i), ALT], h: dir > 0 ? 0 : Math.PI, spin: true, pass: i, x, dir };
      }
      const turn = ease((f - RUN) / (1 - RUN));
      const y = lerp(passY(i), passY(Math.min(PASSES - 1, i + 1)), turn);
      return {
        p: [xs[1] + dir * 25 * Math.sin(turn * Math.PI), y, ALT],
        h: dir > 0 ? turn * Math.PI : Math.PI - turn * Math.PI,
        spin: true,
        done: i + 1,
      };
    }
    const end = [PASSES % 2 ? XB : XA, passY(PASSES - 1)];
    if (t < 17.0) {
      const u = seg(t, TP1, 17.0);
      const q = lerp2(end, [roofV[0], roofV[1]], u);
      return { p: [...q, ALT], h: hdg(end, [roofV[0], roofV[1]]), spin: true };
    }
    const u = seg(t, 17.0, 18.8);
    return { p: [roofV[0], roofV[1], lerp(ALT, roofV[2], u)], h: 0, spin: u < 0.98 };
  }
  // coverage time for a ground point (when the 180 × 120 m footprint first reaches it)
  function coveredAt(x, y) {
    const i0 = Math.max(0, Math.ceil((y - VT.swathM / 2 - Y0) / VT.spacingM));
    if (i0 >= PASSES) return Infinity;
    const dir = i0 % 2 ? -1 : 1;
    const xs = dir > 0 ? [XA, XB] : [XB, XA];
    const xe = x - (dir * VT.footM) / 2;
    const f = clamp01((xe - xs[0]) / (xs[1] - xs[0])) * RUN;
    return TP0 + (i0 + f) * TPD;
  }
  const QROUTE = [...Wd.CREEK].reverse();
  const QSTART = QROUTE[0];
  function quadScan(t) {
    const ALT = QD.altM;
    if (t < 2.2) return { p: roofQ, h: 0, spin: t > 1.6 };
    if (t < 3.6) {
      const u = seg(t, 2.2, 3.6);
      const q = lerp2([roofQ[0], roofQ[1]], QSTART, u);
      return {
        p: [...q, lerp(roofQ[2], ALT, seg(t, 2.2, 3.0))],
        h: hdg([roofQ[0], roofQ[1]], QSTART),
        spin: true,
      };
    }
    if (t < 14.2) {
      const a = along(QROUTE, lin(t, 3.6, 14.2));
      return { p: [...a.p, ALT], h: a.h, spin: true, corridor: lin(t, 3.6, 14.2) };
    }
    const end = QROUTE[QROUTE.length - 1];
    if (t < 17.2) {
      const u = seg(t, 14.2, 17.2);
      const q = lerp2(end, [roofQ[0], roofQ[1]], u);
      return { p: [...q, ALT], h: hdg(end, [roofQ[0], roofQ[1]]), spin: true, corridor: 1 };
    }
    const u = seg(t, 17.2, 18.4);
    return { p: [roofQ[0], roofQ[1], lerp(ALT, roofQ[2], u)], h: 0, spin: u < 0.98, corridor: 1 };
  }
  // ATV: out of the doors, down the ramp, lidar loop round the oak hill, back to the pad

  // ── hub variant ──
  let variant = "container";
  const VAR = () => VARIANTS[variant];
  function setRoof() {
    roofV = hubW([...VAR().vtolPad, VAR().top]);
    roofQ = hubW([...VAR().quadPad, VAR().top]);
  }
  setRoof();

  // ── carriers: payloads and loads per variant ──
  // container: carrier 0 (door end) keeps its 16 for the restoration job; carrier 1's 16 step off at the hub.
  // trailer: one carrier; its top 8 step off at the hub, the bottom 8 do the job.
  const NS16 = 16;
  const payRow = (row) => Array.from({ length: NS16 }, (_, i) => row[i % 4]);
  let CP = [[], []]; // per-carrier slot payloads, set with the scenario
  const slotPay = (vi, i) => (CP[vi] || [])[i] || "none";
  const atvDock = (i) => [VAR().atvs[i] ?? VAR().atvs[0], VAR().laneY];
  const OUT_A = () => [VAR().L / 2 + VAR().ramp + 1.3, VAR().laneY];
  const PARK_A = () => [VAR().L / 2 + 1.2, -5.0];
  const STAGE_PATH = () => {
    const o = OUT_A();
    return [o, [o[0] + 1.2, o[1] - 0.8], [o[0] + 1.8, o[1] - 3.0]];
  };
  const STAGE_H = () => {
    const p = STAGE_PATH();
    return hdg(p[1], p[2]);
  };
  const rampL = (lx) => rampZ(VAR(), lx);
  // carrier local frame → world, slot positions
  const slotLocal = (i) => {
    const [x, y, layer] = ATV.slots[i];
    return [x, y, ATV.slotZ(layer)];
  };
  const onCarrier = (pos, h, local) => {
    const r = rot(local, h);
    return [pos[0] + r[0], pos[1] + r[1], (pos[2] || 0) + r[2]];
  };

  // ── Deploy timing (per variant) ──
  const TD = () =>
    variant === "container"
      ? {
          wing: 9.2,
          mast: 11.4,
          doors: 13.2,
          ramp: 13.8,
          out: [15.0, 16.6],
          toSpot: 1.5,
          off: 20.2,
          waves: 4,
        }
      : {
          wing: 8.4,
          mast: 10.6,
          doors: 12.4,
          ramp: 13.0,
          out: [14.2],
          toSpot: 1.5,
          off: 17.9,
          waves: 2,
        };
  const DRIVE_OUT = 1.8,
    WAVE = 0.8,
    CLIMB = 1.4;
  function driveOut(i, t, t0) {
    const d = atvDock(i),
      o = OUT_A();
    const lx = lerp(d[0], o[0], seg(t, t0, t0 + DRIVE_OUT));
    return { local: [lx, o[1], rampL(lx)], h: 0 };
  }
  // the carrier whose hexapods step off at the hub, and where it parks
  const STAGER = () => (variant === "container" ? 1 : 0);
  function deployAtv(i, t) {
    const T = TD(),
      t0 = T.out[i];
    if (t < t0) return { local: [...atvDock(i), VAR().floor], h: 0 };
    if (t < t0 + DRIVE_OUT) return driveOut(i, t, t0);
    const u = seg(t, t0 + DRIVE_OUT, t0 + DRIVE_OUT + T.toSpot);
    if (i === STAGER()) {
      const a = along(STAGE_PATH(), u);
      return { p: [...hubW(a.p).slice(0, 2), 0], h: a.h };
    }
    const o = OUT_A(),
      pk = PARK_A();
    return { p: [...lerp2(hubW(o), hubW(pk), u), 0], h: lerp(0, -Math.PI / 2, u) };
  }
  const stageSpot = () => {
    const p = STAGE_PATH();
    return [...hubW(p[2]).slice(0, 2), 0];
  };
  // hexapods stepping off at the hub: slots in waves from the top layer down, then a staging grid
  const offSlots = () =>
    ATV.slots.map((s, i) => i).filter((i) => ATV.slots[i][2] >= ATV.layers - TD().waves);
  function offHex(i, t) {
    const T = TD();
    const [sx, , layer] = ATV.slots[i];
    const wave = ATV.layers - 1 - layer;
    const ts = T.off + wave * WAVE;
    const v = stageSpot(),
      h = STAGE_H();
    if (t < ts) return { carried: true };
    const from = onCarrier(v, h, slotLocal(i));
    const col = sx / 0.48;
    const land = onCarrier(v, h, [col * 1.15, 1.3, 0]);
    const grid = onCarrier(v, h, [col * 1.2, 1.8 + (T.waves - 1 - wave) * 1.3, 0]);
    const hh = h + Math.PI / 2;
    const pay = slotPay(STAGER(), i);
    if (t < ts + CLIMB) {
      const u = (t - ts) / CLIMB;
      const p = lerp3(from, land, ease(u));
      p[2] = from[2] * (1 - ease(seg(u, 0.15, 0.95)));
      return {
        p,
        h: hh,
        unfold: seg(u, 0.3, 0.9),
        moving: u > 0.45,
        gait: t * 9,
        payload: pay,
        mast: 0,
      };
    }
    const walk = Math.max(0.6, dist2(land, grid) / 1.1);
    if (t < ts + CLIMB + walk) {
      const p = lerp2(land, grid, ease((t - ts - CLIMB) / walk));
      return { p: [...p, 0], h: hdg(land, grid), moving: true, gait: t * 9, payload: pay, mast: 0 };
    }
    return {
      p: grid,
      h: hh,
      payload: pay,
      arms: ready(hh),
      mast: pay === "mapping" ? seg(t, ts + CLIMB + walk, ts + CLIMB + walk + 0.8) : 0,
    };
  }
  const offEnd = () => {
    const T = TD();
    return T.off + (T.waves - 1) * WAVE + CLIMB + 0.6 + (1.3 * (T.waves - 1) + 0.6) / 1.1;
  };

  // ── Scan: the empty carrier lidar-scans round the oak hill ──
  const oak0 = Wd.oakRing;
  const oakC = oak0.reduce(
    (a, p) => [a[0] + p[0] / oak0.length, a[1] + p[1] / oak0.length],
    [0, 0],
  );
  const LOOP = oak0.map(([x, y]) => {
    const dx = x - oakC[0],
      dy = y - oakC[1],
      l = Math.hypot(dx, dy);
    return [x + (dx / l) * 18, y + (dy / l) * 18];
  });
  const loopStart = LOOP.reduce(
    (b, p, i) => (dist2(p, hubW([12, 0])) < dist2(LOOP[b], hubW([12, 0])) ? i : b),
    0,
  );
  const LOOP_R = [...LOOP.slice(loopStart), ...LOOP.slice(0, loopStart + 1)];
  const atvRoute = () => [stageSpot().slice(0, 2), ...LOOP_R, stageSpot().slice(0, 2)];
  let ATV_ROUTE = atvRoute();
  function atvScan(t) {
    if (t < 2.6) return { p: stageSpot(), h: STAGE_H() };
    if (t < 16.4) {
      const a = along(ATV_ROUTE, lin(t, 2.6, 16.4));
      return { p: [...a.p, 0], h: a.h, scan: true, lidar: lin(t, 2.6, 16.4) };
    }
    return { p: stageSpot(), h: STAGE_H(), lidar: 1 };
  }

  // ── Operate: scenario-driven. Pullers take the flagged targets to the chipper or the bin; planters put items in ──
  let SC = SCN[opts.init] || SCN.nature,
    L = {},
    TGTS = [],
    TGT = [],
    SPOTS = [];
  const cl = (x, y) => [CELL[0] + x, CELL[1] + y];
  const offs = (p, h, d) => [p[0] + Math.cos(h) * d, p[1] + Math.sin(h) * d];
  function setLayout() {
    const y = SC.layout;
    L = {
      ATV_CELL: cl(...y.atv),
      ATV_H: y.atvH,
      CHIP: cl(...y.chip),
      CHIP_H: y.chipH,
      PILE: cl(...y.pile),
      via: (y.via || []).map((q) => cl(...q)),
      atvVia: (y.atvVia || []).map((q) => cl(...q)),
    };
    const bin = SC.dest === "bin";
    L.HOPPER = offs(L.CHIP, L.CHIP_H + Math.PI, bin ? 0.55 : 1.0);
    L.HOPZ = bin ? 1.02 : 0.95;
    L.FEED = offs(L.HOPPER, L.CHIP_H + Math.PI, bin ? 0.95 : 0.85);
    TGTS = SC.targets;
    TGT = TGTS.map((q) => [q.x, q.y]);
    SPOTS =
      SC.plant.spots === "targets"
        ? TGT.map((q, n) => ({ p: q, n }))
        : SC.plant.spots.map((q) => ({ p: q, n: -1 }));
  }
  const QHOVER = () => [CELL[0] + 0.4, CELL[1] - 1.4, 2.4];
  const WORK = 8;
  const workSlots = () =>
    variant === "container" ? [0, 1, 2, 3, 4, 5, 6, 7] : [8, 9, 10, 11, 12, 13, 14, 15];
  const T_UNLOAD = 3.5;
  const PK = () => [0, 4, 2].slice(0, SC.plant.item ? SC.plant.planters : 0);
  const ROLE = (k) => (PK().includes(k) ? "planter" : "puller");
  function carryPay() {
    const work = (i) => {
      const k = workSlots().indexOf(i);
      return k >= 0 && ROLE(k) === "planter" ? "basket" : "none";
    };
    if (variant === "container")
      return [
        Array.from({ length: NS16 }, (_, i) => work(i)),
        payRow(["mapping", "drone", "basket", "none"]),
      ];
    return [
      Array.from({ length: NS16 }, (_, i) =>
        i < 8 ? ["mapping", "drone", "none", "none"][i % 4] : work(i),
      ),
    ];
  }
  // stand-by spots after stepping off: two rows on the carrier's left
  let STAND = [];
  const standSpots = () =>
    Array.from({ length: WORK }, (_, k) => {
      const [sx] = ATV.slots[k];
      const row = k < 4 ? 1 : 0;
      return onCarrier([...L.ATV_CELL, 0], L.ATV_H, [
        (sx / 0.48) * 1.4 + (row ? 0.35 : -0.35),
        1.6 + row * 1.35,
        0,
      ]);
    }).map((q) => [q[0], q[1]]);
  const standOff = (from, s, d) => {
    const l = dist2(from, s) || 1;
    return [s[0] + ((from[0] - s[0]) / l) * d, s[1] + ((from[1] - s[1]) / l) * d];
  };
  const ready = (h) => [rot([0.5, 0.12, 0.5], h), rot([0.5, -0.12, 0.5], h)];
  let OPS;
  function chipperPlan() {
    // container: carrier 1 arrives with the chipper or bin; trailer: carrier 0 drops its hexapods, swaps payloads at the hub, comes back
    if (variant === "container") return { veh: 1, ready: 4.2 };
    const leave = T_UNLOAD + WAVE + CLIMB + 0.4;
    return { veh: 0, leave, atHub: leave + 2.6, swapped: leave + 3.8, ready: leave + 6.6 };
  }
  function buildOps() {
    const cp = chipperPlan();
    STAND = standSpots();
    const nT = TGT.length;
    const pulledAt = new Array(nT).fill(Infinity),
      chippedAt = new Array(nT).fill(Infinity),
      plantedAt = new Array(SPOTS.length).fill(Infinity);
    const shrubTask = new Array(nT);
    const plans = {};
    const pullers = [...Array(WORK).keys()].filter((k) => ROLE(k) === "puller");
    const planters = [...Array(WORK).keys()].filter((k) => ROLE(k) === "planter");
    const startT = (k) => T_UNLOAD + Math.floor(k / 4) * WAVE + CLIMB + 0.2;
    const left = TGT.map((_, n) => n);
    const queue = pullers.map((k) => ({ k, pos: STAND[k], t: startT(k), steps: [] }));
    let hopperFree = cp.ready;
    const spread = pullers.length > 6 ? 0.42 : 0.55,
      FR = Math.max(1.7, 1.1 / spread);
    const fan = (k) => {
      const a = L.CHIP_H + Math.PI + (pullers.indexOf(k) - (pullers.length - 1) / 2) * spread;
      return [L.HOPPER[0] + Math.cos(a) * FR, L.HOPPER[1] + Math.sin(a) * FR];
    };
    while (left.length && queue.length) {
      queue.sort((a, b) => a.t - b.t);
      const q = queue[0];
      left.sort((a, b) => dist2(q.pos, TGT[a]) - dist2(q.pos, TGT[b]));
      const n = left.shift();
      const stand = standOff(q.pos, TGT[n], 0.72);
      const walk = Math.max(0.7, Math.min(2.6, dist2(q.pos, stand) / 0.9));
      const s = { n, from: q.pos, stand, t0: q.t, tw: q.t + walk, h: hdg(stand, TGT[n]) };
      s.tr = s.tw + 0.5;
      s.tg = s.tr + 0.3;
      s.tp = s.tg + 1.0;
      s.tl = s.tp + 0.35;
      s.feedAt = fan(q.k);
      s.carry = Math.max(0.8, Math.min(4, dist2(stand, s.feedAt) / 0.75));
      s.arrive = s.tl + s.carry;
      s.fs = Math.max(s.arrive, hopperFree);
      s.ff = s.fs + 0.5;
      s.fe = s.ff + 0.9;
      hopperFree = s.fe + 0.35;
      s.fh = hdg(s.feedAt, L.HOPPER);
      pulledAt[n] = s.tl;
      chippedAt[n] = s.fe;
      shrubTask[n] = s;
      q.steps.push(s);
      q.t = s.fe + 0.15;
      q.pos = L.FEED;
    }
    queue.forEach((q) => (plans[q.k] = { kind: "puller", steps: q.steps, home: STAND[q.k] }));
    // planters: each spot opens once anything flagged on or beside it is gone; greedy earliest-finish
    const gate = SPOTS.map((sp) => {
      if (sp.n >= 0) return pulledAt[sp.n] + 0.9;
      let g = 0;
      TGT.forEach((q, n) => {
        if (dist2(q, sp.p) < 1.3) g = Math.max(g, pulledAt[n] + 0.9);
      });
      return g;
    });
    const remain = SPOTS.map((_, i) => i);
    const pq = planters.map((k) => ({ k, pos: STAND[k], t: startT(k), steps: [] }));
    while (remain.length && pq.length) {
      pq.sort((a, b) => a.t - b.t);
      const q = pq[0];
      let best = remain[0],
        bestT = Infinity;
      for (const i of remain) {
        const st = standOff(q.pos, SPOTS[i].p, 0.62);
        const walk = Math.max(0.5, dist2(q.pos, st) / 1.0);
        const fin = Math.max(q.t, gate[i] - walk) + walk;
        if (fin < bestT - 1e-9) {
          bestT = fin;
          best = i;
        }
      }
      remain.splice(remain.indexOf(best), 1);
      const sp = SPOTS[best],
        stand = standOff(q.pos, sp.p, 0.62),
        walk = Math.max(0.5, dist2(q.pos, stand) / 1.0);
      const t0 = Math.max(q.t, gate[best] - walk);
      const st = {
        i: best,
        from: q.pos,
        stand,
        t0,
        tw: t0 + walk,
        te: t0 + walk + 1.1,
        h: hdg(stand, sp.p),
      };
      plantedAt[best] = st.te;
      q.steps.push(st);
      q.t = st.te + 0.1;
      q.pos = stand;
    }
    pq.forEach((q) => (plans[q.k] = { kind: "planter", steps: q.steps }));
    const fin = (a) => a.filter(Number.isFinite);
    const lastWork = Math.max(0, ...fin(plantedAt), ...fin(chippedAt));
    return {
      cp,
      pulledAt,
      chippedAt,
      plantedAt,
      shrubTask,
      plans,
      lastWork,
      end: lastWork + 1.4,
      firstPull: Math.min(...pulledAt),
      firstFeed: Math.min(...chippedAt) - 1.0,
      firstPlant: Math.min(...plantedAt),
    };
  }
  const holdPt = (p, h) => {
    const r = rot([0.62, 0, 0], h);
    return [p[0] + r[0], p[1] + r[1]];
  };
  function hexRaw(k, t) {
    const slot = workSlots()[k];
    const pay = slotPay(0, slot);
    const base = { payload: pay, lit: true };
    const wave = Math.floor(k / 4);
    const tu = T_UNLOAD + wave * WAVE;
    const homeH = L.ATV_H + Math.PI / 2,
      CH = L.CHIP_H;
    if (t < tu) return { carried: true };
    if (t < tu + CLIMB) {
      const u = (t - tu) / CLIMB;
      const from = onCarrier([...L.ATV_CELL, 0], L.ATV_H, slotLocal(slot)),
        to = [...STAND[k], 0];
      const p = lerp3(from, to, ease(u));
      p[2] = from[2] * (1 - ease(seg(u, 0.15, 0.95)));
      return { ...base, p, h: homeH, unfold: seg(u, 0.3, 0.9), moving: u > 0.45, gait: t * 9 };
    }
    const plan = OPS.plans[k];
    const walkTo = (a, b, t0, t1, h) => ({
      ...base,
      p: lerp2(a, b, ease((t - t0) / (t1 - t0))),
      h: h ?? hdg(a, b),
      moving: true,
      gait: t * 9,
    });
    if (!plan) return { ...base, p: STAND[k], h: homeH, arms: ready(homeH) };
    if (plan.kind === "planter") {
      const tot = plan.steps.length || 1;
      const loadAt = (tt) =>
        Math.ceil((6 * (tot - plan.steps.filter((s) => s.te <= tt).length)) / tot);
      for (const s of plan.steps) {
        if (t >= s.te + 0.1 && s !== plan.steps[plan.steps.length - 1]) continue;
        if (t < s.t0)
          return { ...base, p: s.from, h: s.h, arms: ready(s.h), basketLoad: loadAt(t) };
        if (t < s.tw) return { ...walkTo(s.from, s.stand, s.t0, s.tw), basketLoad: loadAt(t) };
        const tgt = SPOTS[s.i].p;
        if (t < s.te) {
          const u = (t - s.tw) / (s.te - s.tw);
          const k2 = Math.sin(u * Math.PI);
          return {
            ...base,
            p: s.stand,
            h: s.h,
            basketLoad: loadAt(t),
            reach: twin(tgt, s.h, 0.06 + 0.14 * (1 - k2), k2, u < 0.6 ? 1 : 0),
          };
        }
        return { ...base, p: s.stand, h: s.h, arms: ready(s.h), basketLoad: loadAt(t) };
      }
      return { ...base, p: STAND[k], h: homeH, arms: ready(homeH), basketLoad: loadAt(t) };
    }
    const steps = plan.steps;
    for (let j = 0; j < steps.length; j++) {
      const s = steps[j];
      if (t >= s.fe + 0.15 && j < steps.length - 1) continue;
      const tgt = TGT[s.n];
      if (t < s.t0)
        return { ...base, p: s.from, h: hdg(s.from, s.stand), arms: ready(hdg(s.from, s.stand)) };
      if (t < s.tw) return walkTo(s.from, s.stand, s.t0, s.tw);
      if (t < s.tr)
        return { ...base, p: s.stand, h: s.h, reach: twin(tgt, s.h, 0.13, seg(t, s.tw, s.tr), 0) };
      if (t < s.tg)
        return { ...base, p: s.stand, h: s.h, reach: twin(tgt, s.h, 0.13, 1, seg(t, s.tr, s.tg)) };
      if (t < s.tl)
        return {
          ...base,
          p: s.stand,
          h: s.h,
          reach: twin(tgt, s.h, 0.13 + 0.32 * seg(t, s.tg, s.tp), 1, 1),
        };
      // carry: the plant held out in front
      if (t < s.arrive) {
        const r = walkTo(s.stand, s.feedAt, s.tl, s.arrive);
        return { ...r, arms: hold(r.p, r.h) };
      }
      if (t < s.fs) return { ...base, p: s.feedAt, h: s.fh, arms: hold(s.feedAt, s.fh) };
      if (t < s.ff) {
        const r = walkTo(s.feedAt, L.FEED, s.fs, s.ff, CH);
        return { ...r, arms: hold(r.p, r.h) };
      }
      if (t < s.fe) {
        const u = (t - s.ff) / (s.fe - s.ff);
        const hp = [L.HOPPER[0], L.HOPPER[1], L.HOPZ];
        const hd = holdPt(L.FEED, CH);
        const w = lerp3([hd[0], hd[1], 0.45], hp, ease(u));
        return { ...base, p: L.FEED, h: CH, arms: twinAbs(w, CH, u > 0.85 ? 0 : 1) };
      }
      // done: next plant, or walk home
      const nx = steps[j + 1];
      const back = Math.max(0.8, dist2(L.FEED, plan.home) / 0.9);
      if (nx) return { ...base, p: L.FEED, h: CH, arms: ready(CH) };
      if (t < s.fe + back) return walkTo(L.FEED, plan.home, s.fe, s.fe + back);
      return { ...base, p: plan.home, h: homeH, arms: ready(homeH) };
    }
    return { ...base, p: STAND[k], h: homeH, arms: ready(homeH) };
  }
  // two gripper targets either side of a point (world), with blend k toward it from the ready pose
  function twin(tgt, h, z, k2, grip) {
    const lat = [-Math.sin(h) * 0.07, Math.cos(h) * 0.07];
    return [
      { target: [tgt[0] + lat[0], tgt[1] + lat[1], z], grip, k: k2 },
      { target: [tgt[0] - lat[0], tgt[1] - lat[1], z], grip, k: k2 },
    ];
  }
  function twinAbs(w, h, grip) {
    const lat = [-Math.sin(h) * 0.07, Math.cos(h) * 0.07];
    return [
      { target: [w[0] + lat[0], w[1] + lat[1], w[2]], grip, abs: true },
      { target: [w[0] - lat[0], w[1] - lat[1], w[2]], grip, abs: true },
    ];
  }
  const hold = (p, h) => {
    const hp = holdPt(p, h);
    return twinAbs([hp[0], hp[1], 0.45], h, 1).map((a2) => ({ ...a2, body: true }));
  };
  // carriers in Operate
  const destPay = () => (SC.dest === "bin" ? "bin" : "chipper");
  function atvOperate(i, t) {
    const cp = OPS.cp;
    const A = L.ATV_CELL,
      dA = [Math.cos(L.ATV_H), Math.sin(L.ATV_H)];
    const preA = [A[0] - dA[0] * 4, A[1] - dA[1] * 4 + 0.4 * Math.abs(dA[0])];
    const toCell = (from, t0, t1) => {
      if (t < t0) return { p: [...from, 0], h: hdg(from, A) };
      const path = [from, ...L.atvVia, preA, A];
      const a = along(path, seg(t, t0, t1));
      return { p: [...a.p, 0], h: a.h };
    };
    const dC = [Math.cos(L.CHIP_H), Math.sin(L.CHIP_H)];
    const chipPath = (from) => [
      from,
      ...L.via,
      [L.CHIP[0] - dC[0] * 2.5, L.CHIP[1] - dC[1] * 2.5],
      L.CHIP,
    ];
    const loaded = variant === "container" ? NS16 : 8;
    const dp = destPay();
    const fill = (tt) => OPS.chippedAt.filter((v) => tt > v - 0.4).length / Math.max(1, TGT.length);
    const destState = (tt) => (dp === "chipper" ? { chip: chipping(tt) } : { fill: fill(tt) });
    if (i === 0) {
      const start = variant === "container" ? hubW(PARK_A()) : stageSpot().slice(0, 2);
      if (variant === "container" || t < cp.leave) {
        const r = toCell(start, 0.3, 3.0);
        if (t >= 3.0) {
          r.p = [...A, 0];
          r.h = L.ATV_H;
        }
        return { ...r, payload: "carrier", load: loaded, vi: 0 };
      }
      // trailer: back to the hub, swap to the chipper or bin, return
      const hubSpot = stageSpot().slice(0, 2);
      if (t < cp.atHub) {
        const a = along([A, preA, ...[...L.atvVia].reverse(), hubSpot], seg(t, cp.leave, cp.atHub));
        return { p: [...a.p, 0], h: a.h, payload: "carrier", load: 0, vi: 0 };
      }
      if (t < cp.swapped)
        return {
          p: [...hubSpot, 0],
          h: STAGE_H(),
          payload: t < (cp.atHub + cp.swapped) / 2 ? "carrier" : dp,
          load: 0,
          vi: 0,
        };
      const a = along(chipPath(hubSpot), seg(t, cp.swapped, cp.ready));
      return {
        p: [...a.p, 0],
        h: t >= cp.ready ? L.CHIP_H : a.h,
        payload: dp,
        ...destState(t),
        vi: 0,
      };
    }
    const a = along(chipPath(stageSpot().slice(0, 2)), seg(t, 0.4, cp.ready));
    return {
      p: [...a.p, 0],
      h: t >= cp.ready ? L.CHIP_H : a.h,
      payload: dp,
      ...destState(t),
      vi: 1,
    };
  }
  const chipping = (t) => OPS.shrubTask.some((s) => s && t > s.fs + 0.25 && t < s.fe + 0.5);

  // ── Deploy: hook-lift truck (container) or pickup + trailer ──
  const PATH = (() => {
    const ctrl = [
      [24, 90],
      [22, 45],
      [21, 16],
      [18, 5],
      [12, 0.6],
      [6, 0],
      [0, 0],
      [-20, 0],
      [-80, 0],
    ];
    const pts = [];
    for (let i = 0; i < ctrl.length - 1; i++) {
      const p0 = ctrl[Math.max(0, i - 1)],
        p1 = ctrl[i],
        p2 = ctrl[i + 1],
        p3 = ctrl[Math.min(ctrl.length - 1, i + 2)];
      for (let s = 0; s < 16; s++) {
        const u = s / 16,
          u2 = u * u,
          u3 = u2 * u;
        pts.push(
          [0, 1].map(
            (c) =>
              0.5 *
              (2 * p1[c] +
                (-p0[c] + p2[c]) * u +
                (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * u2 +
                (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * u3),
          ),
        );
      }
    }
    pts.push(ctrl[ctrl.length - 1]);
    const s = [0];
    for (let i = 1; i < pts.length; i++) s.push(s[i - 1] + dist2(pts[i - 1], pts[i]));
    return { pts, s, total: s[s.length - 1] };
  })();
  function pathAt(sv) {
    const { pts, s } = PATH;
    sv = Math.max(0, Math.min(PATH.total - 1e-6, sv));
    let i = 1;
    while (s[i] < sv) i++;
    const f = (sv - s[i - 1]) / (s[i] - s[i - 1] || 1);
    return { p: lerp2(pts[i - 1], pts[i], f), h: hdg(pts[i - 1], pts[i]) };
  }
  const arcAtX = (x) => {
    const { pts, s } = PATH;
    for (let i = 1; i < pts.length; i++)
      if (pts[i][1] < 0.05 && pts[i][0] <= x) {
        const a = pts[i - 1],
          b = pts[i];
        return s[i - 1] + ((a[0] - x) / (a[0] - b[0] || 1)) * (s[i] - s[i - 1]);
      }
    return PATH.total;
  };
  const BED_Z = 1.14,
    BOX_BEHIND = 1.63,
    TOW = 6.89;
  function deployModel(t) {
    const V0 = VAR();
    const out = { vehicles: null, landed: false };
    if (variant === "container") {
      const sStop = arcAtX(1.0) + BOX_BEHIND;
      const sT =
        t < 5.0
          ? lerp(-12, sStop, 1 - Math.pow(1 - clamp01((t - 1.0) / 4.0), 2))
          : t < 6.0
            ? sStop
            : t < 8.2
              ? sStop + 4.2 * seg(t, 6.0, 8.2)
              : sStop + 4.2 + 70 * Math.pow(lin(t, 8.2, 11), 1.6);
      const tr = pathAt(sT);
      let box = null,
        arm = 0;
      if (t < 5.0) {
        const b = pathAt(sT - BOX_BEHIND);
        box = { c: [b.p[0], b.p[1], BED_Z + V0.H / 2], h: b.h + Math.PI, p: 0 };
      } else if (t < 6.0) {
        const u = seg(t, 5.0, 6.0);
        box = { c: [1.0 + 0.45 * u, 0, lerp(BED_Z + V0.H / 2, 2.13, u)], h: 0, p: 0.3 * u };
        arm = u;
      } else if (t < 8.2) {
        const u = seg(t, 6.0, 8.2);
        box = {
          c: [lerp(1.45, 0, u), 0, lerp(2.13, V0.base + V0.H / 2, u)],
          h: 0,
          p: 0.3 * (1 - u),
        };
        arm = 1 - u;
      }
      out.landed = t >= 8.2;
      out.box = box;
      out.truck = t < 11 ? { kind: "hook", p: tr.p, h: tr.h, arm } : null;
    } else {
      const sStop = arcAtX(0) + TOW;
      const sP =
        t < 6.0
          ? lerp(-14, sStop, 1 - Math.pow(1 - clamp01((t - 1.0) / 5.0), 2))
          : t < 7.0
            ? sStop
            : t < 7.6
              ? sStop + 0.6 * seg(t, 7.0, 7.6)
              : sStop + 0.6 + 70 * Math.pow(lin(t, 7.6, 10.5), 1.6);
      const pk = pathAt(sP);
      if (t < 6.0) {
        const b = pathAt(sP - TOW);
        out.box = { c: [b.p[0], b.p[1], V0.base + V0.H / 2], h: b.h + Math.PI, trailer: true };
      }
      out.landed = t >= 6.0;
      out.jacks = seg(t, 6.0, 7.0);
      out.truck = t < 10.5 ? { kind: "pickup", p: pk.p, h: pk.h } : null;
    }
    return out;
  }

  // ── model: everything at (stage, t) ──
  function model(name, t) {
    const m = { name, t };
    const si = ORDER.indexOf(name);
    m.cam = camAt(name, t);
    m.globe = si === 0 ? 1 - seg(t, 4.7, 5.3) : 0;
    m.map = si === 0 ? seg(t, 4.5, 5.0) : 1;
    m.tilt = si === 0 ? lin(t, 5.1, 7.0) : 1;
    m.world = si === 0 ? seg(t, 7.0, 7.9) : 1;
    m.boundary = si === 0 ? seg(t, 10.2, 12.4) : 1;
    m.zones = si === 0 ? seg(t, 13.4, 14.2) : 1;
    m.site = si === 0 ? seg(t, 16.6, 17.4) : si === 1 ? 1 - seg(t, 2.0, 3.0) : 0;
    const T = TD();
    m.dep = si === 1 ? deployModel(t) : null;
    m.hubOn = si >= 2 || (si === 1 && m.dep.landed);
    m.wing = si === 1 ? seg(t, T.wing, T.wing + 1.4) : si > 1 ? 1 : 0;
    m.mast = si === 1 ? seg(t, T.mast, T.mast + 1.4) : si > 1 ? 1 : 0;
    m.dish = si === 1 ? seg(t, T.mast + 0.4, T.mast + 1.2) : si > 1 ? 1 : 0;
    m.online = si > 1 || (si === 1 && t > T.mast + 1.4);
    m.doors = si === 1 ? seg(t, T.doors, T.doors + 1.0) : si > 1 ? 1 : 0;
    m.ramp = si === 1 ? seg(t, T.ramp, T.ramp + 1.0) : si > 1 ? 1 : 0;
    m.jacks = si === 1 ? (variant === "trailer" ? m.dep.jacks : 1) : 1;
    m.hubLit = si === 1 && m.hubOn && t < T.out[0];
    m.fleet = m.hubOn;
    const nAtv = VAR().atvs.length;
    const full = (i) => ({ payload: "carrier", load: NS16, vi: i });
    const staged = variant === "container" ? 0 : NS16 - offSlots().length;
    m.staged = [];
    m.hex = [];
    if (si === 1) {
      m.vtol = { p: roofV, h: 0 };
      m.quad = { p: roofQ, h: 0 };
      m.atvs = [0, 1].slice(0, nAtv).map((i) => ({ ...deployAtv(i, t), ...full(i) }));
      const st = m.atvs[STAGER()];
      st.skip = [];
      m.staged = offSlots().map((i, j) => {
        const r = withOff(offHex(i, t), OFF_DEP, j, t);
        if (!r.carried) st.skip.push(i);
        return r;
      });
    } else if (si === 2) {
      m.vtol = vtolAt(t);
      m.quad = quadScan(t);
      m.atvs =
        variant === "container"
          ? [
              { p: [...hubW(PARK_A()), 0], h: -Math.PI / 2, ...full(0) },
              { ...atvScan(t), payload: "carrier", load: 0, vi: 1 },
            ]
          : [{ ...atvScan(t), payload: "carrier", load: staged, vi: 0 }];
      m.staged = offSlots().map((i) => offHex(i, 99));
    } else if (si === 3) {
      m.vtol = { p: roofV, h: 0 };
      const QH = (tt) =>
        SC.quad
          ? SC.quad(tt)
          : (() => {
              const q = QHOVER();
              return [q[0] + Math.sin(tt * 0.5) * 0.5, q[1] + Math.cos(tt * 0.5) * 0.4, q[2]];
            })();
      const q3 = QH(3.0),
        qn = QH(t);
      const qp =
        t < 0.6
          ? roofQ
          : t < 3.0
            ? lerp3(roofQ, [q3[0], q3[1], 16], seg(t, 0.6, 3.0))
            : t < 5.4
              ? [...lerp2(q3, qn, seg(t, 3.0, 5.4)), lerp(16, qn[2], seg(t, 3.0, 5.4))]
              : qn;
      m.quad = { p: qp, h: t * 0.2, spin: t > 0.4 };
      m.atvs = [atvOperate(0, t), ...(nAtv > 1 ? [atvOperate(1, t)] : [])];
      m.hex = Array.from({ length: WORK }, (_, k) => hexOperate(k, t));
      const c0 = m.atvs[0];
      if (c0.payload === "carrier") c0.skip = workSlots().filter((_, k) => !m.hex[k].carried);
      m.pile =
        SC.dest === "chipper"
          ? OPS.chippedAt.filter((v) => t > v - 0.5).length / Math.max(1, TGT.length)
          : 0;
    } else {
      m.vtol = { p: roofV, h: 0 };
      m.quad = { p: roofQ, h: 0 };
      m.atvs = [0, 1]
        .slice(0, nAtv)
        .map((i) => ({ local: [...atvDock(i), VAR().floor], h: 0, ...full(i) }));
    }
    m.lit = {
      vtol: si === 2 && t < 16,
      quad: si === 2 && t > 2 && t < 16,
      atv: si === 2 && t > 2.6 && t < 16.4,
    };
    m.scanner = si === 2 ? (variant === "container" ? 1 : 0) : -1;
    // broom targets: flagged after the scan passes over them, then pulled, carried, chipped, replaced by oaks
    m.shrubs = TGT.map((p, i) => {
      if (si < 2) return { flag: 0 };
      if (si === 2) return { flag: seg(t, coveredAt(...p) + 0.2, coveredAt(...p) + 0.8) };
      const s = OPS.shrubTask[i];
      if (t >= s.fe) return { gone: true };
      if (t < s.tg) return { flag: 1 };
      if (t < s.tl)
        return {
          lift: 0.32 * seg(t, s.tg, s.tp),
          tip: 0.15 * seg(t, s.tg, s.tp),
          dir: [Math.cos(s.h), Math.sin(s.h)],
        };
      // held in front of the hexapod, lying forward
      const hx = hexOperate(OPS_PULLER_OF[i], t);
      const hp = holdPt(hx.p, hx.h);
      let off = [hp[0] - p[0], hp[1] - p[1], 0],
        shrink = 0,
        tip = lerp(0.15, 1, seg(t, s.tl, s.tl + 0.5)),
        lift = 0.42;
      if (t >= s.ff) {
        const u = ease((t - s.ff) / (s.fe - s.ff));
        const tgt = [L.HOPPER[0] - p[0], L.HOPPER[1] - p[1], L.HOPZ - 0.45];
        off = lerp3(off, tgt, u);
        shrink = u;
        lift = lerp(0.42, 0.4, u);
      }
      return { lift, tip, shrink, offset: off, dir: [Math.cos(hx.h), Math.sin(hx.h)], held: true };
    });
    m.scen = SC.id;
    m.plants = SPOTS.map((sp, i) => ({
      p: sp.p,
      i,
      mulch: si === 3 ? seg(t, OPS.plantedAt[i] - 0.7, OPS.plantedAt[i] - 0.2) : 0,
      g: si === 3 ? seg(t, OPS.plantedAt[i] - 0.45, OPS.plantedAt[i]) : 0,
    }));
    m.done = si === 3 ? seg(t, OPS.lastWork + 0.2, OPS.lastWork + 1.0) : 0;
    m.laptop = si === 2 ? seg(t, 2.5, 3.6) : si === 3 ? 1 - seg(t, 0, 0.6) : 0;
    return m;
  }
  let OPS_PULLER_OF = [];
  setLayout();
  CP = carryPay();
  // ── crowd: a precomputed pass that keeps walkers apart and out of vehicles and trunks ──
  // Each robot keeps its planned path; an offset (≤1.4 m, eased) steers it round others. Robots standing at a task
  // have right of way; robots climbing on or off a carrier are exempt.
  const BODY = 1.08,
    OBST = 0.42;
  function solveCrowd(n, rawAt, t0, t1, dynAt, stat = []) {
    const DT = 0.04,
      steps = Math.max(2, Math.ceil((t1 - t0) / DT) + 1);
    const tab = Array.from({ length: n }, () => new Float32Array(steps * 2));
    // sample every robot's planned state once on the grid; look-ahead reads further along it
    const H = [0, 8, 18, 30],
      G = [1, 0.7, 0.5, 0.3],
      total = steps + H[3] + 1;
    const X = new Float32Array(total * n),
      Y = new Float32Array(total * n),
      LIVE = new Uint8Array(total * n),
      IMM = new Uint8Array(steps * n),
      PIN = new Uint8Array(steps * n);
    for (let st = 0; st < total; st++) {
      const t = t0 + st * DT;
      for (let k = 0; k < n; k++) {
        const r = rawAt(k, t),
          q = st * n + k;
        if (r && !r.carried && r.p) {
          LIVE[q] = 1;
          X[q] = r.p[0];
          Y[q] = r.p[1];
        }
        if (st < steps && r) {
          IMM[q] = r.unfold !== undefined ? 1 : 0;
          PIN[q] = r.moving ? 0 : 1;
        }
      }
    }
    const ox = new Float64Array(n),
      oy = new Float64Array(n),
      px = new Float64Array(n),
      py = new Float64Array(n);
    const maxStep = 2.4 * DT,
      B2 = BODY * BODY;
    for (let st = 0; st < steps; st++) {
      const t = t0 + st * DT,
        b0 = st * n;
      for (let k = 0; k < n; k++) {
        px[k] = ox[k];
        py[k] = oy[k];
        if (!LIVE[b0 + k]) {
          ox[k] = oy[k] = 0;
          continue;
        }
        const f = Math.exp(-DT / (PIN[b0 + k] ? 0.3 : 0.9));
        ox[k] *= f;
        oy[k] *= f;
      }
      const dyn = dynAt(t);
      for (let it = 0; it < 6; it++) {
        for (let hz = 0; hz < 4; hz++) {
          const bb = (st + H[hz]) * n,
            gain = G[hz];
          for (let i = 0; i < n; i++) {
            if (!LIVE[b0 + i] || !LIVE[bb + i]) continue;
            for (let j = i + 1; j < n; j++) {
              if (!LIVE[b0 + j] || !LIVE[bb + j] || (IMM[b0 + i] && IMM[b0 + j])) continue;
              let dx = X[bb + i] + ox[i] - X[bb + j] - ox[j],
                dy = Y[bb + i] + oy[i] - Y[bb + j] - oy[j];
              const d2 = dx * dx + dy * dy;
              if (d2 >= B2) continue;
              let d = Math.sqrt(d2);
              if (d < 1e-4) {
                dx = Math.cos(i + j);
                dy = Math.sin(i + j);
                d = 1e-4;
              } else {
                dx /= d;
                dy /= d;
              }
              const ov = (BODY - d) * gain;
              const wi = IMM[b0 + i] ? 0 : PIN[b0 + i] ? 0.15 : 1,
                wj = IMM[b0 + j] ? 0 : PIN[b0 + j] ? 0.15 : 1,
                sum = wi + wj || 1;
              ox[i] += dx * ov * (wi / sum);
              oy[i] += dy * ov * (wi / sum);
              ox[j] -= dx * ov * (wj / sum);
              oy[j] -= dy * ov * (wj / sum);
            }
          }
        }
        for (let k = 0; k < n; k++) {
          if (!LIVE[b0 + k] || IMM[b0 + k]) continue;
          const kx = X[b0 + k],
            ky = Y[b0 + k];
          for (const ob of stat) {
            const qx = kx + ox[k] - ob[0],
              qy = ky + oy[k] - ob[1],
              rr = ob[2] + OBST;
            if (qx > rr || qx < -rr || qy > rr || qy < -rr) continue;
            const d = Math.hypot(qx, qy);
            if (d < rr && d > 1e-4) {
              ox[k] += (qx / d) * (rr - d);
              oy[k] += (qy / d) * (rr - d);
            }
          }
          for (const ob of dyn) {
            // oriented box [x, y, heading, half-length, half-width]
            const qx = kx + ox[k] - ob[0],
              qy = ky + oy[k] - ob[1];
            const c = Math.cos(ob[2]),
              sn = Math.sin(ob[2]),
              lx = qx * c + qy * sn,
              ly = -qx * sn + qy * c;
            const cx = Math.max(-ob[3], Math.min(ob[3], lx)),
              cy = Math.max(-ob[4], Math.min(ob[4], ly));
            let nx = lx - cx,
              ny = ly - cy,
              d = Math.hypot(nx, ny),
              push;
            if (d > 1e-4) {
              if (d >= OBST) continue;
              push = OBST - d;
              nx /= d;
              ny /= d;
            } else {
              const ex = ob[3] - Math.abs(lx),
                ey = ob[4] - Math.abs(ly);
              if (ex < ey) {
                nx = Math.sign(lx) || 1;
                ny = 0;
                push = ex + OBST;
              } else {
                nx = 0;
                ny = Math.sign(ly) || 1;
                push = ey + OBST;
              }
            }
            ox[k] += (nx * c - ny * sn) * push;
            oy[k] += (nx * sn + ny * c) * push;
          }
        }
      }
      for (let k = 0; k < n; k++) {
        const ddx = ox[k] - px[k],
          ddy = oy[k] - py[k],
          dl = Math.hypot(ddx, ddy);
        if (dl > maxStep) {
          ox[k] = px[k] + (ddx / dl) * maxStep;
          oy[k] = py[k] + (ddy / dl) * maxStep;
        }
        const l = Math.hypot(ox[k], oy[k]);
        if (l > 1.4) {
          ox[k] *= 1.4 / l;
          oy[k] *= 1.4 / l;
        }
        tab[k][st * 2] = ox[k];
        tab[k][st * 2 + 1] = oy[k];
      }
    }
    return { t0, DT, steps, tab };
  }
  function withOff(r, C, k, t) {
    if (!C || !r || r.carried || !r.p) return r;
    const f = Math.max(0, Math.min(C.steps - 1.001, (t - C.t0) / C.DT)),
      i = Math.floor(f),
      u = f - i,
      a = C.tab[k];
    const ox = a[i * 2] * (1 - u) + a[i * 2 + 2] * u,
      oy = a[i * 2 + 1] * (1 - u) + a[i * 2 + 3] * u;
    if (Math.abs(ox) + Math.abs(oy) < 1e-4) return r;
    const out = {
      ...r,
      p: r.p.length === 3 ? [r.p[0] + ox, r.p[1] + oy, r.p[2]] : [r.p[0] + ox, r.p[1] + oy],
    };
    if (r.arms)
      out.arms = r.arms.map((a2) =>
        a2 && a2.body
          ? { ...a2, target: [a2.target[0] + ox, a2.target[1] + oy, a2.target[2]] }
          : a2,
      );
    return out;
  }
  const carObs = (p, h) => (p ? [[p[0], p[1], h, 1.04, 0.86]] : []);
  let OFF_OP = null,
    OFF_DEP = null;
  function hexOperate(k, t) {
    return withOff(hexRaw(k, t), OFF_OP, k, t);
  }
  function solveOperate() {
    OFF_OP = null;
    const stat = SC.obstacles || [];
    OFF_OP = solveCrowd(
      WORK,
      (k, t) => hexRaw(k, t),
      0,
      OPS.end + 2,
      (t) =>
        [0, 1].flatMap((i) => {
          if (i === 1 && variant !== "container") return [];
          const a = atvOperate(i, t);
          return a.p && !a.local ? carObs(a.p, a.h) : [];
        }),
      stat,
    );
  }
  function solveDeploy() {
    OFF_DEP = null;
    const list = offSlots(),
      T = TD();
    OFF_DEP = solveCrowd(
      list.length,
      (j, t) => offHex(list[j], t),
      T.off,
      offEnd() + 1,
      () => carObs(stageSpot(), STAGE_H()),
    );
  }
  // plans and crowd passes are cached per scenario and hub type, so switching back and forth costs nothing
  const OPS_CACHE = new Map(),
    DEP_CACHE = new Map();
  function rebuildOps() {
    const key = SC.id + "|" + variant,
      hit = OPS_CACHE.get(key);
    if (hit) {
      ({ OPS, OPS_PULLER_OF, OFF_OP, STAND } = hit);
      OFF_DEP = DEP_CACHE.get(variant);
      return;
    }
    OPS = buildOps();
    OPS_PULLER_OF = new Array(TGT.length);
    for (const [k, pl] of Object.entries(OPS.plans))
      if (pl && pl.kind === "puller") pl.steps.forEach((s) => (OPS_PULLER_OF[s.n] = Number(k)));
    solveOperate();
    if (opts.lazyDeploy) OFF_DEP = null;
    else if (!DEP_CACHE.has(variant)) {
      solveDeploy();
      DEP_CACHE.set(variant, OFF_DEP);
    } else OFF_DEP = DEP_CACHE.get(variant);
    OPS_CACHE.set(key, { OPS, OPS_PULLER_OF, OFF_OP, STAND });
  }
  rebuildOps();

  // ── callouts: one label at a time, a leader line to where it is happening; held long enough to read ──
  const holdFor = (txt) => Math.max(2.8, Math.min(4.4, 1.3 + txt.length * 0.055));
  function calloutSpec() {
    const T = TD(),
      cp = OPS.cp;
    const vw = (m, i) =>
      m.atvs[i]
        ? m.atvs[i].p
          ? [...m.atvs[i].p.slice(0, 2), 1.4]
          : hubW([...m.atvs[i].local.slice(0, 2), 2])
        : null;
    return {
      plan: [
        [
          7.3,
          "Find your land in the maps app",
          () => [Wd.LAND.x + Wd.LAND.w * 0.62, Wd.LAND.y + Wd.LAND.d * 0.3, 0],
        ],
        [10.3, "Draw the boundary · 160 ha", () => [Wd.LAND.x + Wd.LAND.w, Wd.LAND.y, 0]],
        [13.6, SC.zoneLabel, () => [SC.zoneAt[0], SC.zoneAt[1], 0]],
        [16.9, "Hub site, beside the road", () => [...HUB0, 2]],
      ],
      deploy:
        variant === "container"
          ? [
              [
                1.3,
                "A hook-lift truck delivers the hub",
                (m) => (m.dep.box ? hubW(m.dep.box.c) : [...HUB0, 2]),
              ],
              [
                T.wing + 0.2,
                "Solar wings fold out",
                () => hubW([0.6, VAR().W / 2 + 2.2, VAR().top - 0.3]),
              ],
              [
                T.mast + 1.0,
                "Mast up, Starlink online",
                () => hubW([...VAR().mastAt, VAR().top + 3]),
              ],
              [T.out[0] + 0.6, "Two carriers drive out, 16 hexapods on each", (m) => vw(m, 0)],
              [
                T.off + 0.3,
                "Hexapods unfold and step off",
                (m) => {
                  const h = m.staged.find((x) => !x.carried);
                  return h ? [...h.p.slice(0, 2), 0.4] : null;
                },
              ],
            ]
          : [
              [
                1.3,
                "A pickup tows the hub in",
                (m) => (m.dep.box ? hubW(m.dep.box.c) : [...HUB0, 2]),
              ],
              [
                T.wing + 0.2,
                "Jacks down, solar wings fold out",
                () => hubW([0.4, VAR().W / 2 + 2.0, VAR().top - 0.3]),
              ],
              [
                T.mast + 1.0,
                "Mast up, Starlink online",
                () => hubW([...VAR().mastAt, VAR().top + 3]),
              ],
              [T.out[0] + 0.6, "The carrier drives out with 16 hexapods", (m) => vw(m, 0)],
              [
                T.off + 0.3,
                "Hexapods unfold and step off",
                (m) => {
                  const h = m.staged.find((x) => !x.carried);
                  return h ? [...h.p.slice(0, 2), 0.4] : null;
                },
              ],
            ],
      scan: [
        [0.9, "Drones launch from the roof", (m) => m.vtol.p],
        [4.4, "VTOL maps all 160 ha from 120 m", (m) => m.vtol.p],
        [8.0, "The 3D model builds on your computer", () => Wd.LAPC],
        [11.6, "Quadcopter images the creek up close", (m) => m.quad.p],
        [15.0, "Carrier lidar-scans under the oaks", (m) => vw(m, m.scanner)],
        [18.6, SC.flagLabel, () => [SC.zoneAt[0], SC.zoneAt[1], 0]],
      ],
      operate: [
        [0.4, SC.op.arrive(variant === "container" ? 16 : 8), (m) => vw(m, 0)],
        ...(variant === "container"
          ? [[3.8, `The second carrier brings the ${destPay()}`, (m) => vw(m, 1)]]
          : [[cp.leave + 0.2, `Carrier heads back to swap in the ${destPay()}`, (m) => vw(m, 0)]]),
        [
          5.4,
          SC.op.problem,
          (m) => {
            const i = TGT.findIndex((_, n) => m.shrubs[n].flag);
            return i >= 0 ? [...TGT[i], 0.9] : null;
          },
        ],
        [
          OPS.firstPull - 0.6,
          SC.op.pull,
          (m) => {
            const i = TGT.findIndex((_, n) => m.shrubs[n].held || m.shrubs[n].lift > 0.05);
            if (i < 0) return null;
            const s = m.shrubs[i];
            return [
              TGT[i][0] + (s.offset ? s.offset[0] : 0),
              TGT[i][1] + (s.offset ? s.offset[1] : 0),
              0.8,
            ];
          },
        ],
        ...(variant === "trailer"
          ? [[cp.ready - 0.6, `Back with the ${destPay()}`, (m) => vw(m, 0)]]
          : []),
        [OPS.firstFeed, SC.op.feed, () => [L.HOPPER[0], L.HOPPER[1], 1.0]],
        [OPS.end - 4.2, "Cell done. The crew moves to the next one", () => [...L.ATV_CELL, 1.6]],
        ...(SC.op.plant
          ? [
              [
                OPS.firstPlant - 0.3,
                SC.op.plant,
                (m) => {
                  const p = m.plants.filter((q) => q.g > 0.05).pop();
                  return p ? [...p.p, 0.5] : null;
                },
              ],
            ]
          : []),
        ...(SC.op.extra || []).map(([tt, text, an]) => [
          typeof tt === "function" ? tt(OPS) : tt,
          text,
          an,
        ]),
      ].filter((c) => Number.isFinite(c[0])),
    };
  }
  let CALLOUTS = {};
  function scheduleCallouts() {
    const spec = calloutSpec();
    for (const [stage, list] of Object.entries(spec)) {
      // one label at a time; a second lane opens only when waiting would leave a label more than 1.5 s behind its event
      const free = [0, 0];
      CALLOUTS[stage] = list
        .sort((a, b) => a[0] - b[0])
        .map(([t0, text, anchor]) => {
          let lane = 0;
          if (Math.max(t0, free[0]) - t0 > 1.5 && free[1] < free[0]) lane = 1;
          const s = Math.max(t0, free[lane]);
          const d = holdFor(text);
          free[lane] = s + d + 0.25;
          return { t0: s, t1: s + d, text, anchor, lane };
        });
    }
  }
  scheduleCallouts();
  const stageEnd = (name) => Math.max(...(CALLOUTS[name] || [{ t1: 0 }]).map((c) => c.t1)) + 0.6;

  function routeTo(route, u) {
    const lens = route.slice(1).map((p, i) => dist2(route[i], p));
    let s = clamp01(u) * lens.reduce((a, b) => a + b, 0);
    const out = [route[0]];
    for (let i = 0; i < lens.length; i++) {
      if (s <= lens[i]) {
        out.push(lerp2(route[i], route[i + 1], lens[i] ? s / lens[i] : 0));
        break;
      }
      s -= lens[i];
      out.push(route[i + 1]);
    }
    return out;
  }
  function setDurations() {
    DUR.plan = Math.ceil(stageEnd("plan"));
    DUR.deploy = Math.ceil(Math.max(offEnd(), stageEnd("deploy")));
    DUR.scan = Math.ceil(Math.max(20, stageEnd("scan")));
    DUR.operate = Math.ceil(Math.max(OPS.end, stageEnd("operate")));
  }
  setDurations();
  function setVariant(v) {
    variant = v;
    setRoof();
    ATV_ROUTE = atvRoute();
    CP = carryPay();
    rebuildOps();
    scheduleCallouts();
    setDurations();
  }
  function setScenario(id) {
    if (!SCN[id]) return;
    SC = SCN[id];
    setLayout();
    CP = carryPay();
    rebuildOps();
    scheduleCallouts();
    setDurations();
  }
  const covState = (si, tt) => {
    // swept parcel area (completed passes), current pass band, quad corridor and carrier lidar loop, for painting
    if (si < 2) return null;
    const v = si === 2 ? vtolAt(tt) : { done: PASSES };
    const done =
      si === 3 || tt >= TP1
        ? PASSES
        : v.pass !== undefined
          ? v.pass
          : v.done !== undefined
            ? v.done
            : 0;
    const yTop = Wd.LAND.y,
      yDone = done > 0 ? Math.min(Wd.LAND.y + Wd.LAND.d, passY(done - 1) + VT.swathM / 2) : null;
    let band = null,
      flights = [];
    if (v.pass !== undefined && tt < TP1) {
      const y0 = Math.max(yTop, passY(v.pass) - VT.swathM / 2),
        y1 = Math.min(yTop + Wd.LAND.d, passY(v.pass) + VT.swathM / 2);
      const xa = v.dir > 0 ? Wd.LAND.x : Math.min(Wd.LAND.x + Wd.LAND.w, v.x - VT.footM / 2);
      const xb =
        v.dir > 0 ? Math.min(Wd.LAND.x + Wd.LAND.w, v.x + VT.footM / 2) : Wd.LAND.x + Wd.LAND.w;
      if (xb > xa) band = [xa, y0, xb - xa, y1 - y0];
      flights.push([
        [v.dir > 0 ? XA : XB, passY(v.pass)],
        [v.x, passY(v.pass)],
      ]);
    }
    for (let i = 0; i < Math.min(done, PASSES); i++)
      flights.push([
        [XA, passY(i)],
        [XB, passY(i)],
      ]);
    const LX0 = Wd.LAND.x + 6,
      LX1 = Wd.LAND.x + Wd.LAND.w - 6;
    flights = flights
      .map(([a, b]) => [
        [Math.max(LX0, Math.min(LX1, a[0])), a[1]],
        [Math.max(LX0, Math.min(LX1, b[0])), b[1]],
      ])
      .filter(([a, b]) => Math.abs(b[0] - a[0]) > 1);
    const q = si === 2 ? quadScan(tt) : { corridor: 1 };
    const qBand = q.corridor ? routeTo(QROUTE, q.corridor) : null;
    const a = si === 2 ? atvScan(tt) : { lidar: 1 };
    let aBand = null;
    if (a.lidar) {
      const L0 = dist2(ATV_ROUTE[0], ATV_ROUTE[1]),
        Lt = ATV_ROUTE.slice(1).reduce((s, p, i) => s + dist2(ATV_ROUTE[i], p), 0),
        LL = Lt - dist2(ATV_ROUTE[ATV_ROUTE.length - 2], ATV_ROUTE[ATV_ROUTE.length - 1]) - L0;
      const u = (a.lidar * Lt - L0) / LL;
      if (u > 0) aBand = routeTo(LOOP_R, Math.min(1, u));
    }
    return {
      rect: yDone ? [Wd.LAND.x, yTop, Wd.LAND.w, yDone - yTop] : null,
      band,
      flights,
      qBand,
      aBand,
      done,
    };
  };
  return {
    ORDER,
    DUR,
    model,
    setVariant,
    setScenario,
    setCam: (f) => (camAt = f),
    get variant() {
      return variant;
    },
    get CALLOUTS() {
      return CALLOUTS;
    },
    get OPS() {
      return OPS;
    },
    get SC() {
      return SC;
    },
    get L() {
      return L;
    },
    get TGTS() {
      return TGTS;
    },
    get TGT() {
      return TGT;
    },
    get PILE() {
      return L.PILE;
    },
    get HOPPER() {
      return L.HOPPER;
    },
    get ATV_CELL() {
      return L.ATV_CELL;
    },
    get CARRY_PAY() {
      return { [variant]: CP };
    },
    VAR,
    TD,
    hubW,
    coveredAt,
    vtolAt,
    covState,
    NS16,
    ready,
    holdPt,
    PASSES,
    TP0,
    TP1,
    VT,
    QD,
    QROUTE,
    LOOP_R,
    oakC,
    ease,
    seg,
    lin,
    lerp,
    clamp01,
    dist2,
    mmss: (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`,
  };
}

// Home hero: one landscape, seven crews working at once, a camera gliding between them.
// Each crew is its own story instance anchored at its own work cell; scenery around each is a compact vignette.

function startHero(Wd, OPT) {
  const { canvas, themeEl, zones, route, labels, still } = OPT;
  const E = makeEngine(canvas, { themeEl, maxPixels: OPT.maxPixels() });
  // keep the work areas readable: drop full-size world trees that would stand over a crew
  Wd = {
    ...Wd,
    TREES: Wd.TREES.filter((t) =>
      zones.every((z) => Math.hypot(t.x - z.c[0], t.y - z.c[1]) > 26 + (z.clear || 0)),
    ),
  };
  const world = buildWorld(E, Wd);
  world.screen.visible = false;
  const ground = world.ground;
  const HUB0 = Wd.HUB0;

  // ── crews ──
  const crews = zones.map((z, i) => {
    const W2 = { ...Wd, CELL: z.c };
    const SCN = makeScenarios(W2).S;
    const S = makeStory(W2, ground, SCN, { init: z.id, lazyDeploy: true });
    buildScenes(E, W2, SCN, ground, {
      only: [z.id],
      near: { c: z.c, r: 17, poleR: 70, wide: z.wide },
    }).show(z.id);
    const end = S.OPS.end;
    return {
      z,
      S,
      SC: SCN[z.id],
      end,
      t: 5 + (end - 5) * (z.phase ?? (i * 0.37) % 1),
      done: 0,
      seen: false,
    };
  });

  // ── the base, parked open by the road (static) ──
  const HK = makeKit(E, { n: 5000 }),
    HL = makeLines(E, { dynamic: true, max: 6000 });
  function drawHub() {
    HK.begin();
    HL.begin();
    hub(HK, HL, [HUB0[0], HUB0[1], 0], {
      variant: "container",
      wings: 1,
      mast: 1,
      dish: 1,
      online: true,
      doors: 1,
      ramp: 1,
      jacks: 1,
      lit: false,
      contents: () => undefined,
    });
    HK.end();
    HL.end();
  }
  drawHub();

  // ── per-frame drawing ──
  const D = makeKit(E, { n: 22000 });
  const DL = makeLines(E, { dynamic: true, max: 30000 });
  const toLocalArms = (hx, S) => {
    const base = [hx.p[0], hx.p[1], 0];
    let arms = hx.arms
      ? hx.arms.map((a) =>
          a && a.abs
            ? { target: [a.target[0] - base[0], a.target[1] - base[1], a.target[2]], grip: a.grip }
            : Array.isArray(a)
              ? { target: a, grip: 0.4 }
              : a,
        )
      : null;
    if (hx.reach) {
      const rd = S.ready(hx.h);
      arms = hx.reach.map((a, k) => {
        const off = [a.target[0] - base[0], a.target[1] - base[1], a.target[2]];
        return { target: rd[k].map((v, i) => v * (1 - a.k) + off[i] * a.k), grip: a.grip };
      });
    }
    return arms ? arms.map((a) => (a ? { target: rot(a.target, -hx.h), grip: a.grip } : a)) : null;
  };
  const dashedRing = (c, r, z, n = 20) => {
    for (let i = 0; i < n; i++) {
      const a0 = (i / n) * Math.PI * 2,
        a1 = ((i + 0.6) / n) * Math.PI * 2;
      DL.seg(
        [c[0] + Math.cos(a0) * r, c[1] + Math.sin(a0) * r, z],
        [c[0] + Math.cos(a1) * r, c[1] + Math.sin(a1) * r, z],
        "lit",
      );
    }
  };
  function drawCrew(cr, tt) {
    const { S, SC } = cr;
    const m = S.model("operate", tt);
    for (const a of m.atvs) {
      if (a.local || !a.p) continue;
      carrier(D, [a.p[0], a.p[1], ground(a.p[0], a.p[1])], {
        heading: a.h,
        payload: a.payload || "carrier",
        load: a.load ?? 16,
        skip: a.skip,
        slotPayloads: S.CARRY_PAY[S.variant][a.vi ?? 0],
        chip: a.chip,
        fill: a.fill,
        chipPhase: tt * 1.4,
      });
    }
    for (const hx of m.hex) {
      if (!hx || hx.carried) continue;
      const p = hx.p.length === 2 ? [hx.p[0], hx.p[1], ground(hx.p[0], hx.p[1])] : hx.p;
      hexapod(D, p, {
        heading: hx.h,
        moving: hx.moving,
        gait: hx.gait,
        unfold: hx.unfold ?? 1,
        arms: toLocalArms(hx, S),
        payload: hx.payload,
        basketLoad: hx.basketLoad ?? 6,
        mast: 1,
        lit: 1,
      });
    }
    quad(D, m.quad.p, { heading: m.quad.h, lit: 0 }, DL);
    const TG = S.TGTS;
    m.shrubs.forEach((s2, i) => {
      const T = TG[i];
      if (!T || s2.gone) return;
      const o = s2.offset
        ? [T.x + s2.offset[0], T.y + s2.offset[1], s2.offset[2] || 0]
        : [T.x, T.y, 0];
      shrub(D, o, {
        kind: T.kind,
        seed: T.seed,
        scale: T.scale,
        tip: s2.tip || 0,
        lift: s2.lift || 0,
        shrink: s2.shrink || 0,
        dir: s2.dir || [1, 0],
        lit: s2.flag > 0.5 ? 1 : 0,
      });
      if (s2.flag > 0.5) dashedRing([T.x, T.y], 0.62 * T.scale + 0.25, 0.02);
    });
    const item = SC.plant.item;
    for (const pl of m.plants || []) {
      const [x, y] = pl.p;
      if (item === "oak") {
        if (pl.mulch > 0.01) mulchRing(D, [x, y, 0], 0.48 * pl.mulch);
        if (pl.g > 0.01) sapling(D, [x, y, 0], 0.4 + 0.6 * pl.g);
      } else if (item === "conifer") {
        if (pl.mulch > 0.01) mulchRing(D, [x, y, 0], 0.3 * pl.mulch);
        if (pl.g > 0.01) conifer(D, [x, y, 0], 0.4 + 0.6 * pl.g);
      } else if (item === "plug") {
        if (pl.mulch > 0.01) mulchRing(D, [x, y, 0], 0.18 * pl.mulch);
        if (pl.g > 0.01) plug(D, [x, y, 0], 0.4 + 0.6 * pl.g, pl.i * 1.7);
      } else if (item === "seed") {
        if (pl.mulch > 0.01) mulchRing(D, [x, y, 0], 0.5 * pl.mulch);
        if (pl.g > 0.01) seedPatch(D, [x, y, 0], pl.g, pl.i + 1);
      } else if (item === "wattle") {
        if (pl.g > 0.01) wattle(D, [x, y, 0], 2.0, 0, pl.g);
        if (m.done > 0.01) seedPatch(D, [x, y + 0.55, 0], m.done * 0.45, pl.i + 3);
      }
    }
    if (m.pile > 0.01) mulchRing(D, [...S.PILE, 0], 0.25 + 0.75 * Math.sqrt(m.pile));
    if (SC.id === "infra" && tt > 5.4) {
      const P0 = SC.POLE;
      for (const tp of POLE_TIPS([P0[0], P0[1], 0], SC.LN)) DL.seg(m.quad.p, tp, "lo");
    }
    if (SC.id === "fuels" && m.done > 0.5) dashedRing(SC.HOUSE.c, 9.1, 0.03, 44);
    if (SC.id === "solar" && tt > 6.4) {
      const q = m.quad.p,
        y = SC.PV[1];
      for (const [dx, dy, z] of [
        [-1.1, -1.0, 1.6],
        [1.1, -1.0, 1.6],
        [-1.1, 1.0, 0.75],
        [1.1, 1.0, 0.75],
      ])
        DL.seg(q, [q[0] + dx, y + dy, z], "lo");
    }
    if (SC.id === "pipeline" && tt > 5.6) {
      const q = m.quad.p;
      for (const [dx, dy] of [
        [-0.9, -0.6],
        [0.9, -0.6],
        [0.9, 0.6],
        [-0.9, 0.6],
      ])
        DL.seg(q, [q[0] + dx, SC.PIPE_Y + dy, 0.03], "lo");
    }
  }

  // ── camera: a closed spline through the route, slowing near each crew; framing blends toward each crew's preferred angle ──
  const P = route.map((r) => r.c);
  const cr = (p0, p1, p2, p3, u) => {
    const u2 = u * u,
      u3 = u2 * u;
    return [0, 1].map(
      (c) =>
        0.5 *
        (2 * p1[c] +
          (-p0[c] + p2[c]) * u +
          (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * u2 +
          (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * u3),
    );
  };
  const samples = [];
  for (let i = 0; i < P.length; i++)
    for (let s = 0; s < 40; s++)
      samples.push(
        cr(
          P[(i - 1 + P.length) % P.length],
          P[i],
          P[(i + 1) % P.length],
          P[(i + 2) % P.length],
          s / 40,
        ),
      );
  const arc = [0];
  for (let i = 1; i <= samples.length; i++)
    arc.push(
      arc[i - 1] +
        Math.hypot(
          samples[i % samples.length][0] - samples[i - 1][0],
          samples[i % samples.length][1] - samples[i - 1][1],
        ),
    );
  const LOOP = arc[arc.length - 1];
  const at = (s) => {
    s = ((s % LOOP) + LOOP) % LOOP;
    let i = 1;
    while (arc[i] < s) i++;
    const f = (s - arc[i - 1]) / (arc[i] - arc[i - 1] || 1);
    const a = samples[i - 1],
      b = samples[i % samples.length];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  };
  // ── one landscape: tracks that bend around each crew, ground texture, scattered trees and brush ──
  const CLEAR = 15;
  const bend = (q) => {
    let [x, y] = q;
    for (const c of crews) {
      const R = CLEAR + (c.z.clear || 0),
        dx = x - c.z.c[0],
        dy = y - c.z.c[1],
        d = Math.hypot(dx, dy);
      if (d < R) {
        const k = R / (d || 1e-3);
        x = c.z.c[0] + dx * k;
        y = c.z.c[1] + dy * k;
      }
    }
    return [x, y];
  };
  const TR = [];
  for (let d = 0; d < LOOP; d += 1) TR.push(bend(at(d)));
  for (let i = 1; i < TR.length; i++)
    TR[i] = [TR[i - 1][0] * 0.5 + TR[i][0] * 0.5, TR[i - 1][1] * 0.5 + TR[i][1] * 0.5];
  const trackAt = (s) => {
    s = ((s % TR.length) + TR.length) % TR.length;
    const i = Math.floor(s),
      f = s - i,
      a = TR[i],
      b = TR[(i + 1) % TR.length];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
  };
  {
    const SK = makeKit(E, { n: 9000 }),
      SL = makeLines(E);
    SK.begin();
    let seed = 913;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const nearCrew = (x, y, r) =>
      crews.some((c) => Math.hypot(x - c.z.c[0], y - c.z.c[1]) < r + (c.z.clear || 0));
    const nearTrack = (x, y, r) => {
      for (let i = 0; i < TR.length; i += 2)
        if (Math.abs(TR[i][0] - x) < r && Math.abs(TR[i][1] - y) < r) return true;
      return false;
    };
    // tyre tracks
    for (let i = 0; i < TR.length; i++) {
      const a = TR[i],
        b = TR[(i + 1) % TR.length],
        h = Math.atan2(b[1] - a[1], b[0] - a[0]),
        nx = -Math.sin(h) * 0.62,
        ny = Math.cos(h) * 0.62;
      if (i % 3 === 2) continue;
      for (const sg of [1, -1])
        SL.seg(
          [a[0] + nx * sg, a[1] + ny * sg, ground(a[0], a[1]) + 0.02],
          [b[0] + nx * sg, b[1] + ny * sg, ground(b[0], b[1]) + 0.02],
          "faint",
        );
    }
    // grass tufts over the whole working area
    const xs = TR.map((q) => q[0]),
      ys = TR.map((q) => q[1]);
    const X0 = Math.min(...xs) - 45,
      X1 = Math.max(...xs) + 45,
      Y0 = Math.min(...ys) - 40,
      Y1 = Math.max(...ys) + 40;
    for (let n = 0; n < 14000; n++) {
      const x = X0 + rnd() * (X1 - X0),
        y = Y0 + rnd() * (Y1 - Y0),
        z = ground(x, y) + 0.004;
      if (nearCrew(x, y, 9)) continue;
      SL.seg([x, y, z], [x + 0.09, y + 0.04, z], "faint");
      SL.seg([x, y, z], [x + 0.04, y + 0.1, z], "faint");
    }
    // clumps of oaks and firs, brush, rocks — kept off the tracks and out of the work areas
    const place = (r, tries, fn) => {
      for (let n = 0; n < tries; n++) {
        const x = X0 + rnd() * (X1 - X0),
          y = Y0 + rnd() * (Y1 - Y0);
        if (nearCrew(x, y, r + 4) || nearTrack(x, y, r)) continue;
        fn(x, y, ground(x, y));
      }
    };
    place(16, 40, (x, y) => {
      if (nearCrew(x, y, 26) || Math.hypot(x - HUB0[0], y - HUB0[1]) < 22) return;
      const k = 1 + Math.floor(rnd() * 3);
      for (let j = 0; j < k; j++) {
        const a = rnd() * 6.28,
          d = 1.5 + rnd() * 4;
        const px = x + Math.cos(a) * d,
          py = y + Math.sin(a) * d;
        (rnd() < 0.7 ? oak : fir)(SK, [px, py, ground(px, py)], 0.3 + rnd() * 0.16);
      }
    });
    place(
      4,
      260,
      (x, y, z) =>
        Math.hypot(x - HUB0[0], y - HUB0[1]) > 14 &&
        shrub(SK, [x, y, z], {
          kind: rnd() < 0.5 ? "brush" : "broom",
          seed: Math.floor(rnd() * 999),
          scale: 0.7 + rnd() * 0.7,
        }),
    );
    SK.end();
    SL.build();
  }

  function frame(p) {
    let wsum = 0.25,
      az = 40 * 0.25,
      el = 31 * 0.25,
      w = OPT.width() * 1.0 * 0.25,
      slow = 0;
    for (const r of route) {
      const d = Math.hypot(p[0] - r.c[0], p[1] - r.c[1]),
        wt = Math.exp(-((d / 26) ** 2));
      wsum += wt;
      az += (r.az ?? 40) * wt;
      el += (r.el ?? 31) * wt;
      w += OPT.width() * (r.zoom ?? 1) * wt;
      slow = Math.max(slow, wt);
    }
    return { az: az / wsum, el: el / wsum, width: w / wsum, slow };
  }

  // ── viewer drag: look around, eases back ──
  const user = { az: 0, el: 0, idle: 99 };
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    user.idle = 0;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    user.az -= (e.clientX - drag.x) * 0.25;
    user.el = Math.max(-20, Math.min(40, user.el + (e.clientY - drag.y) * 0.15));
    drag = { x: e.clientX, y: e.clientY };
    user.idle = 0;
  });
  const endDrag = () => (drag = null);
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  // ── theme ──
  function syncTheme() {
    const pick = document.documentElement.dataset.theme;
    themeEl.dataset.theme =
      pick === "light" || pick === "dark"
        ? pick
        : matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    E.setTheme();
    drawHub();
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncTheme);
  new MutationObserver(syncTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  syncTheme();

  // ── loop ──
  let W = 1,
    H = 1,
    last = 0,
    s = OPT.startAtPoint != null ? arc[OPT.startAtPoint * 40] - 4 : OPT.startAt || 0,
    raf = 0,
    visible = true,
    slowT = 0,
    lastWall = 0,
    vtA = 0;
  const cam = { target: [0, 0, 0], az: 40, el: 31, width: 50 };
  let camInit = false;
  function resize() {
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, r.width);
    H = Math.max(1, r.height);
    E.resize(W, H);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();
  const SPEED = 1.5;
  let fly = null;
  // fly the camera along the route to a route point (the shorter way round the loop)
  function goTo(i) {
    const tgt = arc[i * 40] - 4;
    let ds = (((tgt - s) % LOOP) + LOOP) % LOOP;
    if (ds > LOOP / 2) ds -= LOOP;
    fly = { s0: s, ds, t0: performance.now(), dur: Math.min(4200, 1400 + Math.abs(ds) * 9) };
  }
  function tick(now) {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    // move along the route, easing down near each crew
    const p0 = at(s),
      f0 = frame(p0);
    if (fly) {
      const u = Math.min(1, (now - fly.t0) / fly.dur),
        e = u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
      s = fly.s0 + fly.ds * e;
      if (u >= 1) fly = null;
    } else if (!still.matches) s += dt * (SPEED + (1 - f0.slow) ** 2 * 6.5);
    const p = at(s),
      ahead = at(s + 6),
      f = frame([(p[0] + ahead[0]) / 2, (p[1] + ahead[1]) / 2]);
    const target = [(p[0] * 2 + ahead[0]) / 3, (p[1] * 2 + ahead[1]) / 3, 1.2];
    const k = camInit ? 1 - Math.exp(-dt / 1.2) : 1;
    camInit = true;
    cam.target = cam.target.map((v, i) => v + (target[i] - v) * k);
    cam.az += (f.az - cam.az) * k;
    cam.el += (f.el - cam.el) * k;
    cam.width = Math.exp(Math.log(cam.width) + (Math.log(f.width) - Math.log(cam.width)) * k);
    user.idle += dt;
    if (user.idle > 3 && !drag) {
      const kk = 1 - Math.exp(-dt / 0.9);
      user.az -= user.az * kk;
      user.el -= user.el * kk;
    }
    const sway = still.matches ? 0 : 3 * Math.sin(now / 10000);
    E.view.target.set(...cam.target);
    E.view.width = cam.width;
    E.view.az = cam.az + sway + user.az;
    E.view.el = Math.max(8, Math.min(80, cam.el + user.el));
    if (OPT.shift) {
      const sh = OPT.shift(W, H),
        A = (E.view.az * Math.PI) / 180,
        Ee = (E.view.el * Math.PI) / 180,
        sx = (sh.x || 0) * E.view.width,
        sv = (sh.y || 0) * E.view.width * (H / W);
      E.view.target.x += -Math.sin(A) * sx + Math.cos(A) * Math.sin(Ee) * sv;
      E.view.target.y += Math.cos(A) * sx + Math.sin(A) * Math.sin(Ee) * sv;
      E.view.target.z -= Math.cos(Ee) * sv;
    }
    // crews: run in real time; a finished crew holds, then restarts once it's out of view
    LOD.far = E.view.width / W > 0.03;
    D.begin();
    DL.begin();
    const vis = [];
    for (const c of crews) {
      const d = Math.hypot(c.z.c[0] - E.view.target.x, c.z.c[1] - E.view.target.y),
        near = d < E.view.width * 0.95 + 18;
      if (!still.matches) {
        c.t += dt;
        if (c.t > c.end + 1) {
          if (!near) c.t = 5;
          else c.t = c.end + 1;
        }
      }
      const tt = still.matches ? c.end : c.t;
      if (near) {
        drawCrew(c, tt);
        vis.push(c);
      }
    }
    // supply carriers running the track between jobs
    for (let j = 0; j < 3; j++) {
      const u = (still.matches ? 0 : now / 1000) * 3.1 + (j * TR.length) / 3,
        a = trackAt(u),
        b = trackAt(u + 1.5),
        h = Math.atan2(b[1] - a[1], b[0] - a[0]);
      if (Math.hypot(a[0] - E.view.target.x, a[1] - E.view.target.y) < E.view.width * 1.2)
        carrier(D, [a[0], a[1], ground(a[0], a[1])], {
          heading: h,
          payload: j === 1 ? "bin" : "cargo",
          load: 0,
          fill: 0.6,
          chipPhase: 0,
        });
    }
    // a mapping drone circling the whole landscape
    vtA += dt * 0.05;
    const vc = route.reduce(
      (a, r) => [a[0] + r.c[0] / route.length, a[1] + r.c[1] / route.length],
      [0, 0],
    );
    const vp = [vc[0] + Math.cos(vtA) * 90, vc[1] + Math.sin(vtA) * 60, 38];
    vtol(D, vp, { heading: vtA + Math.PI / 2, lit: 0 }, DL);
    D.end();
    DL.end();
    E.render();
    if (labels)
      labels(
        crews.map((c) => ({ id: c.z.id, xy: E.toScreen([c.z.c[0], c.z.c[1], c.z.lz ?? 5]), W, H })),
      );
    // slow GPU: lower the supersampling once, quietly
    slowT =
      now - (lastWall || now) > 24 ? slowT + (now - (lastWall || now)) : Math.max(0, slowT - 8);
    lastWall = now;
    if (slowT > 2000 && E.ss() > 1.01 && E.budget() > 2.6e6) {
      E.setBudget(Math.max(2.6e6, E.budget() * 0.75));
      slowT = 0;
    }
    raf = visible ? requestAnimationFrame(tick) : 0;
  }
  new IntersectionObserver((en) => {
    visible = en[en.length - 1].isIntersecting;
    if (visible && !raf) {
      last = 0;
      raf = requestAnimationFrame(tick);
    }
  }).observe(canvas);
  raf = requestAnimationFrame(tick);
  return {
    E,
    crews,
    goTo,
    route,
    track: TR,
    focus: () => cam.target,
    view: () => ({ az: E.view.az, el: E.view.el, width: E.view.width, W, H }),
  };
}

// Fleet cards: each machine as a live 3D turntable. One renderer draws every card in turn and copies
// the frame into that card's own canvas, so five cards cost one WebGL context.
// Default motion is a slow cinematic orbit with a breathing elevation and push-in; drag to turn a card.

function startFleetCards(OPT) {
  const { cards, themeEl, still } = OPT;
  const off = document.createElement("canvas");
  const E = makeEngine(off, { themeEl, maxPixels: OPT.maxPixels() });
  const THREE = E.THREE;
  const V0 = VARIANTS.container;
  const PAYS = Array.from({ length: 16 }, (_, i) => ["basket", "none", "mapping", "drone"][i % 4]);
  // [look-at centre, fit radius, turntable radius] per unit (from the fleet model's framing)
  // [look-at centre, fit radius, turntable radius, height of the machine]
  const FR = {
    hub: [[0.2, 0, 2.9], 5.0, 4.8, 7.4],
    vtol: [[0, 0, 0.16], 1.12, 1.38, 0.45],
    quad: [[0, 0, 0.06], 0.22, 0.27, 0.16],
    atv: [[0, 0, 0.95], 1.3, 1.45, 1.95],
    hex: [[-0.06, 0, 0.3], 0.7, 0.78, 0.62],
  };
  const groups = {};
  const turntable = (K, LN, R) => {
    K.disc(0, 0, R, -R * 0.06, R * 0.06);
    for (let i = 0; i < 72; i++) {
      const a = (i / 72) * Math.PI * 2,
        l = i % 6 ? 0.025 : 0.05;
      LN.seg(
        [Math.cos(a) * R * (1 - l), Math.sin(a) * R * (1 - l), 0.002],
        [Math.cos(a) * R * 0.985, Math.sin(a) * R * 0.985, 0.002],
        i % 6 ? "lo" : "mid",
      );
    }
  };
  const lodWas = LOD.far;
  LOD.far = false;
  for (const c of cards) {
    const g = new THREE.Group();
    g.visible = false;
    E.scene.add(g);
    groups[c.unit] = g;
    const K = makeKit(E, { n: c.unit === "hub" ? 9000 : 3000, parent: g }),
      LN = makeLines(E, { parent: g });
    K.begin();
    turntable(K, LN, FR[c.unit][2]);
    if (c.unit === "hub") {
      const loaded = (k) =>
        V0.atvs.forEach((x) =>
          carrier(k, [x, V0.laneY, V0.floor], { payload: "carrier", load: 16, slotPayloads: PAYS }),
        );
      hub(K, LN, [0, 0, 0], {
        variant: "container",
        wings: 1,
        mast: 1,
        dish: 1,
        online: true,
        doors: 1,
        ramp: 1,
        jacks: 1,
        contents: loaded,
      });
      vtol(K, [...V0.vtolPad, V0.top], {}, LN);
      quad(K, [...V0.quadPad, V0.top], {}, LN);
      person(K, [V0.L / 2 + 1.6, 2.6, 0], Math.PI);
    } else if (c.unit === "vtol") vtol(K, [0, 0, 0], {}, LN);
    else if (c.unit === "quad") quad(K, [0, 0, 0], {}, LN);
    else if (c.unit === "atv")
      carrier(K, [0, 0, 0], { payload: "carrier", load: 16, slotPayloads: PAYS });
    K.end();
    LN.build();
    // the walker gets its own per-frame kit so its arms can move
    if (c.unit === "hex") {
      c.dyn = makeKit(E, { n: 600, parent: g });
    }
  }
  LOD.far = lodWas;
  const drawWalker = (k, t) => {
    const s1 = Math.sin(t * 0.9),
      s2 = Math.sin(t * 0.9 + 1.7);
    const was = LOD.far;
    LOD.far = false;
    k.begin();
    hexapod(k, [0, 0, 0], {
      payload: "basket",
      basketLoad: 6,
      mast: 1,
      arms: [
        {
          target: [0.62 + 0.05 * s1, 0.2, 0.08 + (0.1 * (1 + s1)) / 2],
          grip: 0.3 + (0.3 * (1 + s1)) / 2,
        },
        { target: [0.56, -0.14 + 0.04 * s2, 0.36 + 0.08 * s2], grip: 0.8 },
      ],
    });
    k.end();
    LOD.far = was;
  };

  // theme
  function syncTheme() {
    const pick = document.documentElement.dataset.theme;
    themeEl.dataset.theme =
      pick === "light" || pick === "dark"
        ? pick
        : matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
    E.setTheme();
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncTheme);
  new MutationObserver(syncTheme).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  syncTheme();

  // per-card state, sizes and interaction
  let W = 1,
    H = 1,
    dpr = 1;
  cards.forEach((c, i) => {
    c.ctx = c.canvas.getContext("2d");
    c.az = 30 + i * 47;
    c.vel = 0;
    c.drag = null;
    c.idle = 0;
    c.seen = false;
    c.phase = i * 1.3;
    c.canvas.addEventListener("pointerdown", (e) => {
      c.drag = { x: e.clientX };
      c.canvas.setPointerCapture(e.pointerId);
      c.vel = 0;
    });
    c.canvas.addEventListener("pointermove", (e) => {
      if (!c.drag) return;
      const dx = e.clientX - c.drag.x;
      c.drag.x = e.clientX;
      c.az -= dx * 0.45;
      c.vel = -dx * 0.45 * 60;
      c.idle = 0;
    });
    const end = () => {
      c.drag = null;
      c.idle = 0;
    };
    c.canvas.addEventListener("pointerup", end);
    c.canvas.addEventListener("pointercancel", end);
    new IntersectionObserver((en) => {
      c.vis = en[en.length - 1].isIntersecting;
      if (c.vis) kick();
    }).observe(c.canvas);
  });
  function size() {
    const r = cards[0].canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width));
    H = Math.max(1, Math.round(r.height));
    dpr = Math.min(2, devicePixelRatio || 1);
    E.resize(W, H);
    for (const c of cards) {
      c.canvas.width = Math.round(W * dpr);
      c.canvas.height = Math.round(H * dpr);
    }
  }
  new ResizeObserver(size).observe(cards[0].canvas);
  size();

  let raf = 0,
    last = 0,
    t = 0;
  function tick(now) {
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 1 / 60;
    last = now;
    t += still.matches ? 0 : dt;
    let any = false;
    for (const c of cards) {
      if (!c.vis) continue;
      any = true;
      // cinematic default: slow orbit, a breathing elevation and a gentle push-in; a drag takes over, then eases back
      c.idle += dt;
      if (!c.drag) {
        if (Math.abs(c.vel) > 1) {
          c.az += c.vel * dt;
          c.vel *= Math.exp(-dt / 0.4);
        } else if (!still.matches && c.idle > 1.5)
          c.az += 7 * dt * Math.min(1, (c.idle - 1.5) / 1.5);
      }
      const [ctr, R] = FR[c.unit],
        ph = t * 0.21 + c.phase;
      for (const u in groups) groups[u].visible = u === c.unit;
      if (c.dyn) drawWalker(c.dyn, t);
      const push = still.matches ? 1 : 1 - 0.06 * (0.5 + 0.5 * Math.sin(ph * 0.8));
      E.view.target.set(ctr[0], ctr[1], ctr[2]);
      E.view.az = c.az;
      E.view.el = 24 + (still.matches ? 0 : 6 * Math.sin(ph));
      // fit both ways: the turntable across, and the machine's height plus the tilted turntable up and down
      const el = (E.view.el * Math.PI) / 180,
        hgt = FR[c.unit][3],
        vExt = hgt * Math.cos(el) + 2 * FR[c.unit][2] * Math.sin(el);
      E.view.width = Math.max(2 * R * 1.35, vExt * (W / H) * 1.18) * push;
      E.render();
      c.ctx.clearRect(0, 0, c.canvas.width, c.canvas.height);
      c.ctx.drawImage(off, 0, 0, c.canvas.width, c.canvas.height);
      if (!c.seen) {
        c.seen = true;
        c.canvas.classList.add("on");
      }
    }
    raf = any ? requestAnimationFrame(tick) : 0;
  }
  function kick() {
    if (!raf) {
      last = 0;
      raf = requestAnimationFrame(tick);
    }
  }
  kick();
}

/** The hero over the bundled world data. */
export const startHeroScene = (opt) => startHero(WORLD, opt);
export { startFleetCards };
