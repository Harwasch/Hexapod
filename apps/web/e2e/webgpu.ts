/**
 * For the specs tagged `@webgpu` (playwright.config.ts runs them with software WebGPU): what
 * the browser's WebGPU adapter is, so a test can say it skipped for want of one rather than
 * fail -- a runner without SwiftShader's Vulkan has no adapter at all, and the trial's
 * fallback to WebGL2 is unit-tested (scanRendererHost.test.ts), not something to measure here.
 */

import type { Page } from "@playwright/test";

/** The adapter's vendor and architecture ("google swiftshader"), or null when there is none. */
export function webgpuAdapter(page: Page): Promise<string | null> {
  return page.evaluate(async () => {
    const gpu = (
      navigator as {
        gpu?: {
          requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string } } | null>;
        };
      }
    ).gpu;
    if (!gpu) return null;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      return `${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? "?"}`;
    } catch {
      return null;
    }
  });
}
