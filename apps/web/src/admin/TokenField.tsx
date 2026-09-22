import { KeyRound } from "lucide-react";
import { useState, type SubmitEvent } from "react";

import { GlassButton, GlassField, GlassInput, useFieldId } from "@twin/ui";

import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/**
 * The write token, entered once and kept in this browser.
 *
 * The same store the globe app's `WriteTokenField` writes to (`twin.settings.v1`), so a
 * token entered on either page works on both — they are the same credential for the same
 * API. Not a `VITE_` variable, for the reason A4 recorded: those are inlined into the
 * bundle and served to every visitor, which for a shared write token means publishing it.
 */
export function TokenField({ onClose }: { onClose: () => void }) {
  const stored = useSettings((state) => state.writeToken);
  const setSettings = useSettings((state) => state.set);
  const setPrompt = useUi((state) => state.setWriteTokenPrompt);
  const [value, setValue] = useState(stored);
  const id = useFieldId("admin-write-token");

  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSettings({ writeToken: value.trim() });
    setPrompt(false);
    onClose();
  };

  return (
    <form className="admin__token" onSubmit={submit} data-testid="admin-token-form">
      <GlassField
        label="Write token"
        htmlFor={id}
        hint="Sent as Authorization: Bearer on every write. Kept in this browser's localStorage, which any script on this origin can read — fine for a single-user prototype, not for real accounts."
      >
        <GlassInput
          id={id}
          type="password"
          value={value}
          autoComplete="off"
          placeholder="API_WRITE_TOKEN"
          onChange={(event) => {
            setValue(event.target.value);
          }}
          data-testid="admin-token-input"
        />
      </GlassField>
      <GlassButton
        size="sm"
        type="submit"
        leadingIcon={<KeyRound size={13} aria-hidden="true" />}
        data-testid="admin-token-save"
      >
        Save
      </GlassButton>
    </form>
  );
}
