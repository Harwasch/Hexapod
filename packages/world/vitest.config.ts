import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    include: ["src/**/*.test.ts"],
    /**
     * The motion tests sweep minutes of simulated time over a 212-node rig — 7,200 frames ×
     * 212 nodes × 9 forcing components — which is well past vitest's 5 s default. These are
     * the tests that prove the tree moves, so they are worth the seconds.
     */
    testTimeout: 60_000,
  },
});
