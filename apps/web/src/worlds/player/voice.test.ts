import { describe, expect, it, vi } from "vitest";
import { VoiceInput, type SpeechRecognizer, type SpeechResultEvent } from "./voice";

function recognizer(): SpeechRecognizer {
  return {
    continuous: false,
    interimResults: false,
    lang: "",
    onresult: null,
    onerror: null,
    onend: null,
    start: vi.fn(),
    abort: vi.fn(),
  };
}
function results(items: [string, boolean][], resultIndex = 0): SpeechResultEvent {
  return {
    resultIndex,
    results: items.map(([transcript, isFinal]) => Object.assign([{ transcript }], { isFinal })),
  };
}
function callbacks() {
  return {
    continuous: true,
    language: "en-US",
    onTranscript: vi.fn(),
    onCommand: vi.fn(),
    onError: vi.fn(),
    onEnd: vi.fn(),
  };
}
describe("spoken world commands", () => {
  it("sends only finalized phrases once while allowing repeated intentional commands", () => {
    const voice = new VoiceInput();
    const speech = recognizer();
    const options = callbacks();
    voice.start(speech, options);
    speech.onresult?.(results([["Make lightning", false]]));
    expect(options.onCommand).not.toHaveBeenCalled();
    speech.onresult?.(results([["Make lightning strike the tower", true]]));
    speech.onresult?.(
      results([
        ["Make lightning strike the tower", true],
        ["Again", false],
      ]),
    );
    speech.onresult?.(
      results(
        [
          ["Make lightning strike the tower", true],
          ["Again", true],
        ],
        1,
      ),
    );
    speech.onresult?.(
      results(
        [
          ["Make lightning strike the tower", true],
          ["Again", true],
          ["Again", true],
        ],
        2,
      ),
    );
    expect(options.onCommand.mock.calls).toEqual([
      ["Make lightning strike the tower"],
      ["Again"],
      ["Again"],
    ]);
    expect(speech.continuous).toBe(true);
  });

  it("ignores already queued browser callbacks after stop or a replacement interaction", () => {
    const voice = new VoiceInput();
    const speech = recognizer();
    const options = callbacks();
    voice.start(speech, options);
    const stale = speech.onresult;
    voice.stop();
    stale?.(results([["Late command", true]]));
    expect(options.onCommand).not.toHaveBeenCalled();
    expect(speech.abort).toHaveBeenCalledOnce();
    expect(options.onEnd).toHaveBeenCalledOnce();
    voice.start(recognizer(), options);
    stale?.(results([["Old interaction", true]]));
    expect(options.onCommand).not.toHaveBeenCalled();
  });

  it("cleans up errors without an automatic permission or listening retry", () => {
    const voice = new VoiceInput();
    const speech = recognizer();
    const options = callbacks();
    voice.start(speech, options);
    speech.onerror?.({ error: "not-allowed" });
    expect(options.onError).toHaveBeenCalledWith("not-allowed");
    expect(speech.onresult).toBeNull();
    expect(speech.start).toHaveBeenCalledOnce();
    expect(options.onEnd).toHaveBeenCalledOnce();
  });
});
