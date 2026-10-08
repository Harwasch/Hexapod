import "@testing-library/jest-dom/vitest";
import { MotionGlobalConfig } from "motion/react";
import { afterEach } from "vitest";

import { scanPayloads } from "@/lib/payloadCache";

// Fetched scan payloads are kept for the page's life (lib/payloadCache.ts); each test fetches
// its own.
afterEach(() => scanPayloads.clear());

// jsdom cannot run animations; finish motion transitions instantly.
MotionGlobalConfig.skipAnimations = true;

// jsdom lacks these browser APIs that Radix / motion touch.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => undefined,
      removeListener: () => undefined,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      dispatchEvent: () => false,
    }) as MediaQueryList;
}
if (!window.ResizeObserver) {
  window.ResizeObserver = class {
    observe(): void {
      /* jsdom has no layout */
    }
    unobserve(): void {
      /* noop */
    }
    disconnect(): void {
      /* noop */
    }
  };
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => undefined;
}
