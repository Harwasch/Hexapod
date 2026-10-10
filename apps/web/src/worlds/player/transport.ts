export interface LiveTransport {
  request<T>(path: string, init?: RequestInit): Promise<T>;
  frame(path: string, signal: AbortSignal): Promise<Blob | null>;
}
export interface StreamMetrics {
  codec?: string;
  encoder?: string;
  networkRttMs?: number;
  controlLatencyMs?: number;
  framesDecoded?: number;
  framesDropped?: number;
  workerDroppedFrames?: number;
  packetsLost?: number;
  jitterMs?: number;
  fallbackReason?: string;
}
export type VideoConnection = (() => void) & {
  /** Null means no open channel. Once sent, errors never trigger automatic duplicate HTTP actions. */
  sendAction: (action: object) => Promise<unknown> | null;
};
interface ControlAck {
  type?: string;
  id?: string;
  ok?: boolean;
  error?: string;
  result?: unknown;
  droppedFrames?: number;
  encoder?: string;
}

/** Encoded WebRTC first; authenticated still frames remain an explicit compatibility transport. */
export async function connectVideo(
  api: LiveTransport,
  sessionId: string,
  video: HTMLVideoElement,
  sessionSignal: AbortSignal,
  onStatus: (status: string) => void,
  onFrame: (image: ImageBitmap) => void,
  onMetrics?: (metrics: StreamMetrics) => void,
  options: { audio?: boolean } = {},
): Promise<VideoConnection> {
  const stopController = new AbortController();
  const signal = AbortSignal.any([sessionSignal, stopController.signal]);
  let peer: RTCPeerConnection | undefined;
  let channel: RTCDataChannel | undefined;
  let stopped = false;
  let pollTimer: ReturnType<typeof setTimeout> | undefined;
  let connectTimer: ReturnType<typeof setTimeout> | undefined;
  let statsTimer: ReturnType<typeof setTimeout> | undefined;
  let frameRunning = false;
  let failures = 0;
  let previousHash = "";
  let mediaStream: MediaStream | undefined;
  let measuring = false;
  const pending = new Map<
    string,
    {
      resolve(value: unknown): void;
      reject(error: Error): void;
      timer: ReturnType<typeof setTimeout>;
      started: number;
    }
  >();
  const rejectPending = () => {
    for (const item of pending.values()) {
      clearTimeout(item.timer);
      item.reject(
        new Error(
          "Control connection closed. The last command may have been applied; it was not resent.",
        ),
      );
    }
    pending.clear();
  };
  const cleanup = () => {
    if (stopped) return;
    stopped = true;
    stopController.abort();
    clearTimeout(pollTimer);
    clearTimeout(connectTimer);
    clearTimeout(statsTimer);
    rejectPending();
    channel?.close();
    peer?.close();
    mediaStream?.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    signal.removeEventListener("abort", cleanup);
  };
  const connection: VideoConnection = Object.assign(cleanup, {
    sendAction(action: object): Promise<unknown> | null {
      if (stopped || channel?.readyState !== "open") return null;
      if (channel.bufferedAmount > 65536 || pending.size >= 32)
        return Promise.reject(
          new Error("Control transport is busy. Wait before sending another command."),
        );
      const id = crypto.randomUUID();
      const activeChannel = channel;
      const message = JSON.stringify({ id, action });
      if (new TextEncoder().encode(message).length > 16384)
        return Promise.reject(new Error("Command exceeds the live transport limit."));
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          channel?.close();
          reject(
            new Error(
              "Command acknowledgement timed out. It may have been applied; it was not resent.",
            ),
          );
        }, 5000);
        pending.set(id, { resolve, reject, timer, started: performance.now() });
        try {
          activeChannel.send(message);
        } catch {
          clearTimeout(timer);
          pending.delete(id);
          reject(new Error("Command could not be sent."));
        }
      });
    },
  });
  signal.addEventListener("abort", cleanup, { once: true });
  if (signal.aborted) {
    cleanup();
    return connection;
  }

  async function poll() {
    if (stopped || signal.aborted) return;
    try {
      const blob = await api.frame(`/sessions/${sessionId}/frame`, signal);
      if (blob && !stopped) {
        const hash = Array.from(
          new Uint8Array(await crypto.subtle.digest("SHA-256", await blob.arrayBuffer())),
        ).join(",");
        if (hash !== previousHash) {
          const image = await createImageBitmap(blob);
          if (stopped) image.close();
          else {
            previousHash = hash;
            onFrame(image);
            onStatus("Live · frame transport");
          }
        }
      }
      failures = 0;
    } catch (error) {
      if (stopped || signal.aborted) return;
      failures++;
      if (failures > 2)
        onStatus(`Stream interrupted · ${error instanceof Error ? error.message : "retrying"}`);
    }
    if (!stopped)
      pollTimer = setTimeout(() => void poll(), failures ? Math.min(5000, failures * 750) : 100);
  }
  const fallback = (
    reason = "The WebRTC connection could not be established. Check the worker TURN configuration.",
  ) => {
    if (stopped || frameRunning) return;
    frameRunning = true;
    clearTimeout(connectTimer);
    clearTimeout(statsTimer);
    rejectPending();
    channel?.close();
    peer?.close();
    mediaStream?.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
    onMetrics?.({ fallbackReason: reason });
    onStatus(
      options.audio
        ? "Compatibility image stream · generated audio is unavailable on this transport"
        : "Waiting for generated frames · compatibility transport…",
    );
    void poll();
  };
  async function measure() {
    if (stopped || frameRunning || !peer || measuring) return;
    measuring = true;
    try {
      const report = await peer.getStats();
      const metrics: StreamMetrics = {};
      report.forEach(
        (
          entry: RTCStats & {
            kind?: string;
            codecId?: string;
            framesDecoded?: number;
            framesDropped?: number;
            packetsLost?: number;
            jitter?: number;
            state?: string;
            nominated?: boolean;
            currentRoundTripTime?: number;
          },
        ) => {
          if (entry.type === "inbound-rtp" && entry.kind === "video") {
            const codec = entry.codecId
              ? (report.get(entry.codecId) as { mimeType?: string } | undefined)
              : undefined;
            metrics.codec = codec?.mimeType;
            metrics.framesDecoded = entry.framesDecoded;
            metrics.framesDropped = entry.framesDropped;
            metrics.packetsLost = entry.packetsLost;
            metrics.jitterMs = entry.jitter === undefined ? undefined : entry.jitter * 1000;
          }
          if (
            entry.type === "candidate-pair" &&
            entry.state === "succeeded" &&
            entry.nominated &&
            entry.currentRoundTripTime !== undefined
          )
            metrics.networkRttMs = entry.currentRoundTripTime * 1000;
        },
      );
      if (!stopped && !frameRunning) onMetrics?.(metrics);
    } catch {
      /* Unsupported statistics do not interrupt the stream. */
    }
    measuring = false;
    if (!stopped && !frameRunning) {
      clearTimeout(statsTimer);
      statsTimer = setTimeout(() => void measure(), 2000);
    }
  }
  if (typeof RTCPeerConnection === "undefined") {
    fallback();
    return connection;
  }
  try {
    const config = await api.request<{
      iceServers?: RTCIceServer[];
      iceTransportPolicy?: RTCIceTransportPolicy;
    }>(`/sessions/${sessionId}/transport-config`, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(8000)]),
    });
    if (stopped) return connection;
    peer = new RTCPeerConnection({
      iceServers: config.iceServers ?? [],
      iceTransportPolicy: config.iceTransportPolicy ?? "all",
    });
    peer.addTransceiver("video", { direction: "recvonly" });
    if (options.audio) peer.addTransceiver("audio", { direction: "recvonly" });
    channel = peer.createDataChannel("world-controls-v1", { ordered: true });
    channel.onclose = rejectPending;
    channel.onmessage = (event) => {
      if (typeof event.data !== "string" || event.data.length > 16384 || stopped) return;
      let message: ControlAck;
      try {
        const parsed: unknown = JSON.parse(event.data);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
        message = parsed;
      } catch {
        return;
      }
      if (message.type === "telemetry") {
        onMetrics?.({
          workerDroppedFrames:
            typeof message.droppedFrames === "number" &&
            Number.isFinite(message.droppedFrames) &&
            message.droppedFrames >= 0
              ? message.droppedFrames
              : undefined,
          encoder:
            typeof message.encoder === "string" && message.encoder.length <= 64
              ? message.encoder
              : undefined,
        });
        return;
      }
      if (message.type !== "ack" || typeof message.id !== "string" || !message.id) return;
      const item = pending.get(message.id);
      if (!item) return;
      pending.delete(message.id);
      clearTimeout(item.timer);
      onMetrics?.({ controlLatencyMs: performance.now() - item.started });
      if (message.ok === true) item.resolve(message.result);
      else
        item.reject(
          new Error(
            typeof message.error === "string" ? message.error : "The worker rejected this control.",
          ),
        );
    };
    peer.ontrack = (event) => {
      if (stopped || frameRunning) {
        event.track.stop();
        return;
      }
      mediaStream ??= new MediaStream();
      for (const track of [event.track, ...event.streams.flatMap((stream) => stream.getTracks())])
        if (!mediaStream.getTracks().includes(track)) mediaStream.addTrack(track);
      video.srcObject = mediaStream;
      void video.play().catch(() => onStatus("Stream ready · click the world to enable playback"));
    };
    peer.onconnectionstatechange = () => {
      if (stopped) return;
      if (peer?.connectionState === "connected") {
        clearTimeout(connectTimer);
        clearTimeout(statsTimer);
        onStatus("Live · WebRTC encoded video");
        void measure();
      }
      if (peer?.connectionState === "failed") fallback();
      if (peer?.connectionState === "disconnected") {
        clearTimeout(connectTimer);
        connectTimer = setTimeout(() => {
          if (peer?.connectionState !== "connected") fallback();
        }, 5000);
      }
    };
    await peer.setLocalDescription(await peer.createOffer());
    await new Promise<void>((resolve) => {
      if (peer?.iceGatheringState === "complete" || signal.aborted) return resolve();
      const deadline = setTimeout(done, 8000);
      function done() {
        clearTimeout(deadline);
        peer?.removeEventListener("icegatheringstatechange", check);
        signal.removeEventListener("abort", done);
        resolve();
      }
      function check() {
        if (peer?.iceGatheringState === "complete") done();
      }
      peer?.addEventListener("icegatheringstatechange", check);
      signal.addEventListener("abort", done, { once: true });
    });
    if (stopped) return connection;
    const answer = await api.request<RTCSessionDescriptionInit>(`/sessions/${sessionId}/offer`, {
      method: "POST",
      body: JSON.stringify(peer.localDescription),
      signal: AbortSignal.any([signal, AbortSignal.timeout(28000)]),
    });
    if (stopped) return connection;
    await peer.setRemoteDescription(answer);
    connectTimer = setTimeout(() => {
      if (peer?.connectionState !== "connected") fallback();
    }, 15000);
  } catch (error) {
    fallback(error instanceof Error ? error.message : "WebRTC negotiation failed.");
  }
  return connection;
}
