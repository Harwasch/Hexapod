import { Box, Film, Images, UploadCloud } from "lucide-react";
import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { classify } from "./recipes";

export interface DropZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  busy?: boolean;
}

/** The three things a capture can be, and what each becomes. Video leads: it is the common case. */
const KINDS = [
  { icon: Film, label: "Video", result: "3D model" },
  { icon: Images, label: "Photos", result: "3D model" },
  { icon: Box, label: "Splat or scan", result: "placed as is" },
] as const;

/**
 * Drop target for a capture.
 *
 * The `<input type="file">` inside the label is not decoration: it is the control. A
 * drag-and-drop-only zone can only be driven from a test by building a `DataTransfer`
 * by hand, while a real input is what `setInputFiles` (and a keyboard, and a phone)
 * already know how to use.
 *
 * What was dropped is classified by `recipes.ts` and the proposal is shown right under
 * the zone, so the person sees "1 video → Reconstruct · about 8 minutes" before anything
 * else happens.
 */
export function DropZone({ onFiles, disabled = false, busy = false }: DropZoneProps) {
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<{ summary: string; estimate: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (files.length === 0) return;
    const proposal = classify(files);
    setPreview({ summary: proposal.summary, estimate: proposal.estimate });
    onFiles(files);
    if (input.current) input.current.value = "";
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setOver(false);
    if (disabled) return;
    take(event.dataTransfer.files);
  };

  const onChange = (event: ChangeEvent<HTMLInputElement>) => take(event.target.files);

  return (
    <div className="dropzone-wrap">
      <label
        className={`dropzone ${over ? "dropzone--over" : ""} ${disabled ? "dropzone--disabled" : ""}`}
        data-testid="capture-dropzone"
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <span className="dropzone__head">
          <UploadCloud size={20} aria-hidden="true" />
          <span className="dropzone__title">
            {disabled ? "Uploads need the API" : "Drop files or browse"}
          </span>
        </span>
        <span className="dropzone__kinds" aria-hidden={disabled}>
          {KINDS.map(({ icon: Icon, label, result }) => (
            <span key={label} className="dropzone__kind">
              <Icon size={16} aria-hidden="true" />
              <span className="dropzone__kind-label">{label}</span>
              <span className="dropzone__kind-result">{result}</span>
            </span>
          ))}
        </span>
        <input
          ref={input}
          type="file"
          multiple
          className="sr-only"
          disabled={disabled || busy}
          onChange={onChange}
          data-testid="capture-file-input"
        />
      </label>
      {preview && !disabled && (
        <p className="dropzone__proposal" data-testid="capture-proposal">
          <strong>{preview.summary}</strong>
          <span>{preview.estimate}</span>
        </p>
      )}
    </div>
  );
}
