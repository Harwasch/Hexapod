import { describe, expect, it } from "vitest";
import { medianHeightCamera, nextTrainingCamera, trainingFov } from "@/view/liveCamera";
import type { DecodedCameras } from "@/view/liveData";

describe("training views", () => {
  const cameras: DecodedCameras = {
    count: 5,
    centres: new Float32Array([0, -4, 0, 1, -1, 0, 2, -9, 0, 3, -3, 0, 4, -2, 0]),
    forward: new Float32Array(15),
    up: new Float32Array(15),
  };
  it("starts at the median height along solve up, not the middle path entry", () => {
    expect(medianHeightCamera(cameras, [0, -1, 0])).toBe(3);
    expect(medianHeightCamera({ ...cameras, count: 0 }, [0, 1, 0])).toBeNull();
  });
  it("steps in path order and wraps, without sorting the path by height", () => {
    expect([3, 4, 0, 1, 2].map((i) => nextTrainingCamera(i, 5))).toEqual([4, 0, 1, 2, 3]);
    expect(nextTrainingCamera(null, 5)).toBe(0);
    expect(nextTrainingCamera(null, 0)).toBeNull();
  });
  it("uses the measured vertical FOV with a 50 degree legacy fallback", () => {
    expect(trainingFov(67.2)).toBe(67.2);
    expect(trainingFov()).toBe(50);
    expect(trainingFov(null)).toBe(50);
    expect(trainingFov(NaN)).toBe(50);
  });
});
