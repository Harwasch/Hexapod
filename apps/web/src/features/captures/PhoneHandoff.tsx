import { QrCode, Smartphone, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { GlassButton, Spinner } from "@twin/ui";

import { useCreateHandoff } from "@/api/queries";

/**
 * "Add from phone" — the QR code that gets a capture off the device it is on.
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
        leadingIcon={<QrCode size={14} aria-hidden="true" />}
        onClick={() => {
          setOpen(true);
          handoff.mutate({ captureId });
        }}
      >
        Add from phone
      </GlassButton>
    );
  }

  return (
    <div className="handoff" data-testid="handoff">
      <div className="handoff__head">
        <span className="handoff__title">
          <Smartphone size={15} aria-hidden="true" /> Scan with your phone
        </span>
        <GlassButton
          iconOnly
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

      {handoff.isPending && (
        <div className="glass-row">
          <Spinner label="Making a link" />
          <span className="handoff__note">Making a link…</span>
        </div>
      )}

      {handoff.isError && (
        <p className="card__error" data-testid="handoff-error">
          Couldn’t make a link. {handoff.error.message}
        </p>
      )}

      {data && (
        <>
          <div className="handoff__body">
            {/* The API renders this; it is our own server's SVG, not user content. */}
            <div
              className="handoff__qr"
              data-testid="handoff-qr"
              dangerouslySetInnerHTML={{ __html: data.qrSvg }}
            />
            <ol className="handoff__steps">
              <li>Point the phone’s camera at the code.</li>
              <li>Pick videos, photos or a scan.</li>
              <li>They show up here as they upload.</li>
            </ol>
          </div>
          <p className="handoff__note">
            The link works for 10 minutes and can only add files to this capture.
          </p>
        </>
      )}
    </div>
  );
}
