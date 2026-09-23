import { QrCode, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { GlassButton } from "@twin/ui";

import { useCreateHandoff } from "@/api/queries";

/**
 * "Send from your phone" — the QR code that gets a capture off the device it is on.
 *
 * The SVG is rendered by the API, not here: drawing a QR code is the server's job when
 * the server already has the string, and the alternative is the console growing a QR
 * library to render a picture of something it was handed.
 *
 * The code encodes a *handoff* token, not the write token. It authorises uploads to one
 * capture and expires in ten minutes, so a code photographed over your shoulder buys an
 * attacker the ability to add a file to a capture you just made — and nothing else.
 *
 * `autoOpen` is for "New capture from phone", where the capture exists only to be sent
 * to, so making the person click a second button to see the code would be a step with
 * no decision in it. `onClose` lets that caller drop the panel it is shown in.
 */
export function PhoneHandoff({
  captureId,
  autoOpen = false,
  onClose,
}: {
  captureId: string;
  autoOpen?: boolean;
  onClose?: () => void;
}) {
  const [open, setOpen] = useState(autoOpen);
  const handoff = useCreateHandoff();
  const data = handoff.data;

  // Once per capture, not once per render: a second mint would replace a code someone
  // may already be pointing a phone at. The ref also absorbs StrictMode's double effect.
  const minted = useRef<string | null>(null);
  const { mutate } = handoff;
  useEffect(() => {
    if (autoOpen && minted.current !== captureId) {
      minted.current = captureId;
      mutate({ captureId });
    }
  }, [autoOpen, captureId, mutate]);

  if (!open) {
    return (
      <GlassButton
        size="sm"
        variant="ghost"
        data-testid="handoff-open"
        onClick={() => {
          setOpen(true);
          handoff.mutate({ captureId });
        }}
      >
        <QrCode size={14} aria-hidden="true" /> Send from your phone
      </GlassButton>
    );
  }

  return (
    <div className="handoff" data-testid="handoff">
      <div className="handoff__head">
        <span className="handoff__title">Scan with your phone</span>
        <GlassButton
          size="sm"
          variant="ghost"
          aria-label="Close the phone handoff"
          onClick={() => {
            setOpen(false);
            handoff.reset();
            onClose?.();
          }}
        >
          <X size={14} aria-hidden="true" />
        </GlassButton>
      </div>

      {handoff.isPending && <p className="handoff__note">Making a link…</p>}

      {handoff.isError && (
        <p className="card__error" data-testid="handoff-error">
          Could not make a handoff link. {handoff.error.message}
        </p>
      )}

      {data && (
        <>
          {/* The API renders this; it is our own server's SVG, not user content. */}
          <div
            className="handoff__qr"
            data-testid="handoff-qr"
            dangerouslySetInnerHTML={{ __html: data.qrSvg }}
          />
          <p className="handoff__note">
            Expires in ten minutes. It can only add files to this capture. Keep this panel open: the
            files appear below as they arrive.
          </p>
        </>
      )}
    </div>
  );
}
