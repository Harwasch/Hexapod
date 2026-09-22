/**
 * The data console's entry point.
 *
 * A third Vite entry rather than a route inside the globe app, and the import list is the
 * reason: nothing here reaches `@/cesium`, `@/features` or the scene stores, so the
 * bundler has nothing to pull a 3D globe in through. `e2e/admin.spec.ts` asserts that by
 * watching what the page requests, because a comment cannot fail a build.
 */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./admin.css";

import { AdminApp } from "./AdminApp";

const container = document.getElementById("root");
if (!container) throw new Error("#root element missing");

createRoot(container).render(
  <StrictMode>
    <AdminApp />
  </StrictMode>,
);
