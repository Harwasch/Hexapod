import { UploadCloud } from "lucide-react";
import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { classify } from "./recipes";

export interface DropZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
  busy?: boolean;
}

/**
 * Drop target for a capture.
 *
 * The `<input type="file">` inside the label is not decoration: it is the control. A
 * drag-and-drop-only zone can only be driven from a test by building a `DataTransfer`
 * by hand, while a real input is what `setInputFiles` (and a keyboard, and a phone)
 * already know how to use.
 */
export function DropZone({ onFiles, disabled = false, busy = false }: DropZoneProps) {
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const take = (list: FileList | null) => {
    const files = Array.from(list ?? []);
    if (files.length === 0) return;
    const proposal = classify(files);
    setPreview(`${proposal.summary} · ${proposal.estimate}`);
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
    <div className="glass-stack" style={{ gap: "0.4rem" }}>
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
        <UploadCloud size={22} aria-hidden="true" />
        <span className="dropzone__title">
          {disabled ? "Uploads need the API" : "Drop a capture, or choose files"}
        </span>
        <span className="dropzone__hint">
          {disabled
            ? "The catalog API is offline, so there is nowhere to put the bytes."
            : "Video, photos, or a splat you already have. Large files upload in parts."}
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
          {preview}
        </p>
      )}
    </div>
  );
}
