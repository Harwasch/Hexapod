import "@testing-library/jest-dom/vitest";
import { MotionGlobalConfig } from "motion/react";

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
