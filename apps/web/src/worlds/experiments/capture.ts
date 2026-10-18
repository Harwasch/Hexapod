/** Records sampled delivered frames locally, without microphone or generated audio claims. */
export class ExperimentCapture {
  private canvas = document.createElement("canvas");
  private recorder?: MediaRecorder;
  private stream?: MediaStream;
  private chunks: Blob[] = [];
  private bytes = 0;
  private stopped = false;
  async frame(blob: Blob): Promise<void> {
    if (this.stopped) return;
    const bitmap = await createImageBitmap(blob);
    try {
      if (!this.recorder) {
        this.canvas.width = Math.min(832, bitmap.width);
        this.canvas.height = Math.round((bitmap.height * this.canvas.width) / bitmap.width);
        if (typeof MediaRecorder === "undefined" || !this.canvas.captureStream) return;
        const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(
          (value) => MediaRecorder.isTypeSupported(value),
        );
        if (!mime) return;
        this.stream = this.canvas.captureStream(0);
        this.recorder = new MediaRecorder(this.stream, {
          mimeType: mime,
          videoBitsPerSecond: 2_000_000,
        });
        this.recorder.ondataavailable = (event) => {
          if (this.bytes + event.data.size > 32 * 1024 * 1024) {
            if (this.recorder?.state === "recording") this.recorder.stop();
            this.stopped = true;
            return;
          }
          this.chunks.push(event.data);
          this.bytes += event.data.size;
        };
        this.recorder.start(1000);
      }
      this.canvas.getContext("2d")?.drawImage(bitmap, 0, 0, this.canvas.width, this.canvas.height);
      const track = this.stream?.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined;
      track?.requestFrame();
    } finally {
      bitmap.close();
    }
  }
  async finish(): Promise<Blob | undefined> {
    this.stopped = true;
    if (this.recorder && this.recorder.state !== "inactive") {
      const recorder = this.recorder;
      await new Promise<void>((resolve) => {
        recorder.addEventListener("stop", () => resolve(), { once: true });
        recorder.stop();
      });
    }
    this.stream?.getTracks().forEach((track) => track.stop());
    return this.chunks.length
      ? new Blob(this.chunks, { type: this.recorder?.mimeType ?? "video/webm" })
      : undefined;
  }
}
