import { describe, expect, it } from "vitest";

import { readEnv } from "@/app/env";

describe("readEnv", () => {
  it("parses optional asset ids and flags", () => {
    const env = readEnv({
      VITE_CESIUM_ION_ACCESS_TOKEN: " tok ",
      VITE_DEFAULT_SPLAT_ASSET_ID: "4547222",
      VITE_DEFAULT_MESH_ASSET_ID: "abc",
      VITE_DEFAULT_POINTCLOUD_ASSET_ID: "",
      VITE_ENABLE_PHOTOREALISTIC: "true",
      VITE_ENABLE_DEV_TOOLS: "0",
      DEV: false,
    } as unknown as ImportMetaEnv);
    expect(env.ionAccessToken).toBe("tok");
    expect(env.defaultSplatAssetId).toBe(4547222);
    expect(env.defaultMeshAssetId).toBeUndefined();
    expect(env.defaultPointCloudAssetId).toBeUndefined();
    expect(env.photorealisticEnabled).toBe(true);
    expect(env.devToolsEnabled).toBe(false);
  });

  it("defaults to no token and dev tools on in development", () => {
    const env = readEnv({ DEV: true } as unknown as ImportMetaEnv);
    expect(env.ionAccessToken).toBeUndefined();
    expect(env.devToolsEnabled).toBe(true);
    expect(env.apiBaseUrl).toBe("");
  });
});
