import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { connectVideo, type LiveTransport } from "./transport";

describe("live video transport", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("RTCPeerConnection", undefined);
    vi.stubGlobal("crypto", {
      randomUUID: () => "test-command-id",
      subtle: { digest: vi.fn((_algorithm: string, bytes: ArrayBuffer) => Promise.resolve(bytes)) },
    });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("counts only changed fallback frames and cancels future requests on teardown", async () => {
    const close = vi.fn();
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(() => Promise.resolve({ close })),
    );
    const frame = vi.fn(() =>
      Promise.resolve({
        arrayBuffer: () => Promise.resolve(new Uint8Array([1, 2, 3]).buffer),
      } as Blob),
    );
    const api: LiveTransport = { frame, request: vi.fn() };
    const onFrame = vi.fn();
    const controller = new AbortController();
    const cleanup = await connectVideo(
      api,
      "session-id",
      document.createElement("video"),
      controller.signal,
      vi.fn(),
      onFrame,
    );
    await vi.advanceTimersByTimeAsync(350);
    expect(frame.mock.calls.length).toBeGreaterThan(1);
    expect(onFrame).toHaveBeenCalledTimes(1);
    cleanup();
    const before = frame.mock.calls.length;
    await vi.advanceTimersByTimeAsync(1000);
    expect(frame).toHaveBeenCalledTimes(before);
  });

  it("does not deliver frames that finish decoding after cancellation", async () => {
    const close = vi.fn();
    const controller = new AbortController();
    vi.stubGlobal(
      "createImageBitmap",
      vi.fn(() => {
        controller.abort();
        return Promise.resolve({ close });
      }),
    );
    const frame = vi.fn(() =>
      Promise.resolve({ arrayBuffer: () => Promise.resolve(new Uint8Array([4]).buffer) } as Blob),
    );
    const api: LiveTransport = {
      frame,
      request: vi.fn(),
    };
    const onFrame = vi.fn();
    await connectVideo(
      api,
      "session-id",
      document.createElement("video"),
      controller.signal,
      vi.fn(),
      onFrame,
    );
    await vi.advanceTimersByTimeAsync(100);
    expect(onFrame).not.toHaveBeenCalled();
    expect(close).toHaveBeenCalledOnce();
    expect(frame).toHaveBeenCalledOnce();
  });

  it("aborts an in-flight frame request when closed without aborting its parent session", async () => {
    let frameSignal: AbortSignal | undefined;
    const api: LiveTransport = {
      request: vi.fn(),
      frame: (_path, signal) => {
        frameSignal = signal;
        return new Promise((_resolve, reject) =>
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("Stopped", "AbortError")),
            { once: true },
          ),
        );
      },
    };
    const parent = new AbortController();
    const cleanup = await connectVideo(
      api,
      "session",
      document.createElement("video"),
      parent.signal,
      vi.fn(),
      vi.fn(),
    );
    cleanup();
    expect(frameSignal?.aborted).toBe(true);
    expect(parent.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1000);
  });

  function mockedPeer() {
    const channel = {
      readyState: "open",
      bufferedAmount: 0,
      onmessage: null as ((event: { data: string }) => void) | null,
      onclose: null as (() => void) | null,
      send: vi.fn(),
      close: vi.fn(() => {
        channel.readyState = "closed";
        channel.onclose?.();
      }),
    };
    const peer = {
      connectionState: "new",
      iceGatheringState: "complete",
      localDescription: { type: "offer", sdp: "fixture" },
      onconnectionstatechange: null as (() => void) | null,
      ontrack: null as
        ((event: { track: { stop(): void }; streams: MediaStream[] }) => void) | null,
      close: vi.fn(),
      addTransceiver: vi.fn(),
      createDataChannel: () => channel,
      createOffer: () => Promise.resolve({ type: "offer", sdp: "fixture" }),
      setLocalDescription: () => Promise.resolve(),
      setRemoteDescription: () => Promise.resolve(),
      getStats: vi.fn(() => Promise.resolve(new Map())),
    };
    vi.stubGlobal(
      "RTCPeerConnection",
      vi.fn(function () {
        return peer;
      }),
    );
    return { peer, channel };
  }

  it("rejects pending controls on cleanup and safely ignores malformed messages", async () => {
    const { channel } = mockedPeer();
    const api: LiveTransport = {
      request: vi.fn().mockResolvedValue({ iceServers: [], type: "answer", sdp: "fixture" }),
      frame: vi.fn(),
    };
    const cleanup = await connectVideo(
      api,
      "session",
      document.createElement("video"),
      new AbortController().signal,
      vi.fn(),
      vi.fn(),
    );
    expect(() => channel.onmessage?.({ data: "null" })).not.toThrow();
    expect(() => channel.onmessage?.({ data: "[]" })).not.toThrow();
    const result = cleanup.sendAction({ type: "native", action: "forward" });
    expect(result).not.toBeNull();
    const caught = result?.catch((error: unknown) => error);
    cleanup();
    expect(await caught).toBeInstanceOf(Error);
    expect(channel.send).toHaveBeenCalledOnce();
    expect(cleanup.sendAction({ type: "native", action: "forward" })).toBeNull();
    await vi.advanceTimersByTimeAsync(6000);
  });

  it("negotiates native audio and retains tracks arriving in separate streams", async () => {
    const { peer } = mockedPeer();
    const videoTrack = { kind: "video", stop: vi.fn() };
    const audioTrack = { kind: "audio", stop: vi.fn() };
    class Stream {
      constructor(private tracks: { kind: string; stop: () => void }[] = []) {}
      getTracks() {
        return this.tracks;
      }
      addTrack(track: { kind: string; stop: () => void }) {
        this.tracks.push(track);
      }
    }
    vi.stubGlobal("MediaStream", Stream);
    const video = document.createElement("video");
    vi.spyOn(video, "play").mockResolvedValue();
    const api: LiveTransport = {
      request: vi.fn().mockResolvedValue({ iceServers: [], type: "answer", sdp: "fixture" }),
      frame: vi.fn(),
    };
    const cleanup = await connectVideo(
      api,
      "session",
      video,
      new AbortController().signal,
      vi.fn(),
      vi.fn(),
      undefined,
      { audio: true },
    );
    expect(peer.addTransceiver).toHaveBeenCalledWith("audio", { direction: "recvonly" });
    peer.ontrack?.({
      track: videoTrack,
      streams: [new Stream([videoTrack]) as unknown as MediaStream],
    });
    peer.ontrack?.({
      track: audioTrack,
      streams: [new Stream([audioTrack]) as unknown as MediaStream],
    });
    expect((video.srcObject as MediaStream).getTracks()).toEqual([videoTrack, audioTrack]);
    cleanup();
    expect(videoTrack.stop).toHaveBeenCalledOnce();
    expect(audioTrack.stop).toHaveBeenCalledOnce();
  });

  it("keeps one statistics loop across repeated connected events", async () => {
    const { peer } = mockedPeer();
    const api: LiveTransport = {
      request: vi.fn().mockResolvedValue({ iceServers: [], type: "answer", sdp: "fixture" }),
      frame: vi.fn(),
    };
    const cleanup = await connectVideo(
      api,
      "session",
      document.createElement("video"),
      new AbortController().signal,
      vi.fn(),
      vi.fn(),
    );
    peer.connectionState = "connected";
    peer.onconnectionstatechange?.();
    peer.onconnectionstatechange?.();
    await vi.advanceTimersByTimeAsync(0);
    expect(peer.getStats).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(2000);
    expect(peer.getStats).toHaveBeenCalledTimes(2);
    cleanup();
  });

  it("does not revive a failed video transport when a late track event arrives", async () => {
    const { peer } = mockedPeer();
    const api: LiveTransport = {
      request: vi.fn().mockResolvedValue({ iceServers: [], type: "answer", sdp: "fixture" }),
      frame: vi.fn().mockResolvedValue(null),
    };
    const video = document.createElement("video");
    const cleanup = await connectVideo(
      api,
      "session",
      video,
      new AbortController().signal,
      vi.fn(),
      vi.fn(),
    );
    peer.connectionState = "failed";
    peer.onconnectionstatechange?.();
    const stop = vi.fn();
    peer.ontrack?.({ track: { stop }, streams: [] });
    expect(stop).toHaveBeenCalledOnce();
    expect(video.srcObject).toBeNull();
    cleanup();
  });

  it("times out an unacknowledged control without resending it over HTTP", async () => {
    const { channel } = mockedPeer();
    const request = vi.fn().mockResolvedValue({ iceServers: [], type: "answer", sdp: "fixture" });
    const api: LiveTransport = { request, frame: vi.fn() };
    const cleanup = await connectVideo(
      api,
      "session",
      document.createElement("video"),
      new AbortController().signal,
      vi.fn(),
      vi.fn(),
    );
    const result = cleanup
      .sendAction({ type: "prompt", prompt: "Test fixture only" })
      ?.catch((error: unknown) => error);
    await vi.advanceTimersByTimeAsync(5000);
    const rejection: unknown = await result;
    expect(rejection).toBeInstanceOf(Error);
    expect((rejection as Error).message).toContain("not resent");
    expect(channel.send).toHaveBeenCalledOnce();
    expect(request).toHaveBeenCalledTimes(2);
    expect(cleanup.sendAction({ type: "pause" })).toBeNull();
    cleanup();
  });
});
