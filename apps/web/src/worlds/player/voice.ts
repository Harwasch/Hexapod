export interface SpeechResultEvent {
  results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
  resultIndex: number;
}
export interface SpeechRecognizer {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechResultEvent) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  abort: () => void;
}
export type SpeechWindow = Window & {
  SpeechRecognition?: new () => SpeechRecognizer;
  webkitSpeechRecognition?: new () => SpeechRecognizer;
};

/** Owns one explicit microphone interaction; never restarts listening in the background. */
export class VoiceInput {
  private recognition: SpeechRecognizer | null = null;
  private ended: (() => void) | undefined;

  start(
    recognition: SpeechRecognizer,
    options: {
      continuous: boolean;
      language: string;
      onTranscript: (text: string) => void;
      onCommand: (text: string) => void;
      onError: (message: string) => void;
      onEnd: () => void;
    },
  ) {
    this.stop();
    this.recognition = recognition;
    this.ended = options.onEnd;
    const delivered = new Set<number>();
    recognition.continuous = options.continuous;
    recognition.interimResults = true;
    recognition.lang = options.language;
    recognition.onresult = (event) => {
      if (this.recognition !== recognition) return;
      const transcript: string[] = [];
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result?.[0]?.transcript.trim() ?? "";
        if (text) transcript.push(text);
        if (result?.isFinal && text && !delivered.has(i)) {
          delivered.add(i);
          options.onCommand(text);
        }
      }
      options.onTranscript(transcript.join(" "));
    };
    recognition.onerror = (event) => {
      if (this.recognition !== recognition) return;
      options.onError(event.error);
      this.stop();
    };
    recognition.onend = () => {
      if (this.recognition === recognition) this.detach();
    };
    try {
      recognition.start();
    } catch (error) {
      this.stop();
      throw error;
    }
  }

  private detach() {
    const recognition = this.recognition;
    this.recognition = null;
    if (recognition) {
      recognition.onresult = null;
      recognition.onerror = null;
      recognition.onend = null;
    }
    const ended = this.ended;
    this.ended = undefined;
    ended?.();
    return recognition;
  }

  stop() {
    const recognition = this.detach();
    try {
      recognition?.abort();
    } catch {
      // Some browser implementations throw if recognition has already stopped.
    }
  }
}
