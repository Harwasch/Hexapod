/**
 * The site finder (finder.html): the app's globe, Descent search and parcel picker on a page
 * of their own, opened in a dialog by the landing page's "Find my land" (src/landing/finder.ts)
 * so the landing page itself never loads CesiumJS, and so the app's styles never meet the
 * landing page's. It works on its own at /finder.html too.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@/styles/app.css";
import "@/features/land/land.css";
import "@/features/land/surface/surface.css";
import "./finder.css";

import { AppProviders } from "@/app/providers";

import { Finder } from "./Finder";

const container = document.getElementById("root");
if (!container) throw new Error("#root element missing");

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
createRoot(container).render(
  <StrictMode>
    <AppProviders>
      <Finder reducedMotion={reducedMotion} />
    </AppProviders>
  </StrictMode>,
);
