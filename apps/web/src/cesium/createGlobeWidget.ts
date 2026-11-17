import { CesiumWidget } from "cesium";

/** Shared scene bootstrap for the web application and bundled iOS globe. */
export function createGlobeWidget(host: HTMLElement, credits: HTMLElement): CesiumWidget {
  return new CesiumWidget(host, {
    scene3DOnly: true,
    shouldAnimate: true,
    baseLayer: false,
    // No sky box (and so, from the widget, no sun or moon) yet: deferNightSky adds them once
    // the globe has its first tiles.
    skyBox: false,
    // Render only when something changed (camera, tiles, entities, explicit requests).
    // Every manager calls scene.requestRender() after it mutates the scene.
    requestRenderMode: true,
    // Render errors are logged, toasted and recovered from here; Cesium's own panel would
    // stop the app dead on an engine hiccup.
    showRenderLoopErrors: false,
    maximumRenderTimeChange: Number.POSITIVE_INFINITY,
    // Motion never uses MSAA; the PerformanceManager raises it for the still frame.
    msaaSamples: 1,
    contextOptions: { webgl: { powerPreference: "high-performance", antialias: true } },
    // The "Data attribution" dialog is modal and belongs over the whole app, but Cesium
    // would hang it inside the widget — which lives in `.viewport`, a stacking context
    // pinned at `--z-map`, so the HUD would paint over the dialog and swallow its clicks.
    // Hosting it on <body> takes it out of that context; `app.css` then puts it on the
    // sheet layer and dresses it in the glass material.
    creditViewport: document.body,
    creditContainer: credits,
  });
}
