/**
 * Map tiles that wait out a busy server (cesium/busyTiles.ts): a 503 or 429 is asked again when
 * the server says, a few times; any other failure fails at once, as before.
 */
import { RequestErrorEvent, UrlTemplateImageryProvider } from "cesium";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { BusyAwareImageryProvider, busyWait } from "@/cesium/busyTiles";

const busy = (status = 503, headers = "retry-after: 3\r\n") =>
  new RequestErrorEvent(status, null, headers);

/** A tile request failing as Cesium fails one: with a RequestErrorEvent, not an Error. */
// eslint-disable-next-line @typescript-eslint/prefer-promise-reject-errors
const failWith = (event: RequestErrorEvent) => Promise.reject(event);

describe("what counts as busy", () => {
  it("is a 503 or 429, waiting as long as the server says, within reason", () => {
    expect(busyWait(busy())).toBe(3);
    expect(busyWait(busy(429, "Retry-After: 0\r\n"))).toBe(1);
    expect(busyWait(busy(503, "retry-after: 600\r\n"))).toBe(30);
    // Not said (or a date, which is not read): the USGS tiler's own two seconds.
    expect(busyWait(busy(503, ""))).toBe(2);
    expect(busyWait(busy(503, "retry-after: Wed, 21 Oct 2026 07:28:00 GMT\r\n"))).toBe(2);
  });

  it("is never a missing tile, a server error or an image that would not decode", () => {
    expect(busyWait(busy(404))).toBeNull();
    expect(busyWait(busy(502))).toBeNull();
    expect(busyWait(new Error("decode failed"))).toBeNull();
  });
});

describe("a busy tile", () => {
  let provider: BusyAwareImageryProvider;
  beforeEach(() => {
    vi.useFakeTimers();
    provider = new BusyAwareImageryProvider({ url: "https://tiles.test/{z}/{x}/{y}.png" });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("is asked again after the server's Retry-After, and arrives", async () => {
    const image = { width: 256 } as unknown as ImageBitmap;
    const fetch = vi
      .spyOn(UrlTemplateImageryProvider.prototype, "requestImage")
      .mockReturnValueOnce(failWith(busy()))
      .mockReturnValueOnce(Promise.resolve(image));
    const tile = provider.requestImage(1, 2, 3);
    await vi.advanceTimersByTimeAsync(2_900);
    expect(fetch).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(2_000);
    await expect(tile).resolves.toBe(image);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1]?.slice(0, 3)).toEqual([1, 2, 3]);
  });

  it("waits for the browser's own queue to that server without counting it as a try", async () => {
    const image = { width: 256 } as unknown as ImageBitmap;
    const fetch = vi
      .spyOn(UrlTemplateImageryProvider.prototype, "requestImage")
      .mockReturnValueOnce(failWith(busy(503, "retry-after: 1\r\n")))
      .mockReturnValueOnce(undefined)
      .mockReturnValueOnce(undefined)
      .mockReturnValueOnce(undefined)
      .mockReturnValueOnce(Promise.resolve(image));
    const tile = provider.requestImage(1, 2, 3);
    await vi.advanceTimersByTimeAsync(3_000);
    await expect(tile).resolves.toBe(image);
    expect(fetch).toHaveBeenCalledTimes(5);
  });

  it("is not asked again once its layer has left the globe", async () => {
    const fetch = vi
      .spyOn(UrlTemplateImageryProvider.prototype, "requestImage")
      .mockImplementationOnce(() => failWith(busy(503, "retry-after: 1\r\n")))
      .mockReturnValue(undefined);
    const waiting = provider.requestImage(1, 2, 3);
    const failed = expect(waiting).rejects.toBeInstanceOf(RequestErrorEvent);
    // Another tile waits in the browser's queue to the server when the layer goes.
    await vi.advanceTimersByTimeAsync(2_000);
    const asked = fetch.mock.calls.length;
    expect(asked).toBeGreaterThan(1);
    provider.cancel();
    await vi.advanceTimersByTimeAsync(30_000);
    await failed;
    expect(fetch).toHaveBeenCalledTimes(asked);
    // A tile waiting out the server's Retry-After when the layer goes gives up without asking.
    const other = new BusyAwareImageryProvider({ url: "https://tiles.test/{z}/{x}/{y}.png" });
    fetch.mockReset();
    fetch.mockImplementation(() => failWith(busy(503, "retry-after: 3\r\n")));
    const late = other.requestImage(4, 5, 6);
    const gaveUp = expect(late).rejects.toBeInstanceOf(RequestErrorEvent);
    await vi.advanceTimersByTimeAsync(1_000);
    other.cancel();
    await vi.advanceTimersByTimeAsync(30_000);
    await gaveUp;
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("fails after a few tries, and a tile that is not busy fails at once", async () => {
    const fetch = vi
      .spyOn(UrlTemplateImageryProvider.prototype, "requestImage")
      .mockImplementation(() => failWith(busy(503, "retry-after: 1\r\n")));
    const tile = provider.requestImage(1, 2, 3);
    const failed = expect(tile).rejects.toBeInstanceOf(RequestErrorEvent);
    await vi.advanceTimersByTimeAsync(10_000);
    await failed;
    expect(fetch).toHaveBeenCalledTimes(4);

    fetch.mockReset();
    fetch.mockReturnValueOnce(failWith(busy(404)));
    await expect(provider.requestImage(4, 5, 6)).rejects.toMatchObject({ statusCode: 404 });
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
