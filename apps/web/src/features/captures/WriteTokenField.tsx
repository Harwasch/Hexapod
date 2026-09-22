import { KeyRound } from "lucide-react";
import { useState, type SubmitEvent } from "react";

import { GlassButton, GlassField, GlassInput, useFieldId } from "@twin/ui";

import { useSettings } from "@/state/settings";
import { useUi } from "@/state/ui";

/**
 * Appears only after a write came back `401`.
 *
 * A deployment with no `API_WRITE_TOKEN` leaves writes open, so asking for a token up
 * front would invent a step that most runs of this app do not have. The token is stored
 * with the other preferences rather than in a `VITE_` variable, which would publish it
 * in the bundle to everyone who loads the page.
 */
export function WriteTokenField({ onSaved }: { onSaved: () => void }) {
  const stored = useSettings((s) => s.writeToken);
  const setSettings = useSettings((s) => s.set);
  const setPrompt = useUi((s) => s.setWriteTokenPrompt);
  const [value, setValue] = useState(stored);
  const id = useFieldId("write-token");

  const submit = (event: SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = value.trim();
    if (!token) return;
    setSettings({ writeToken: token });
    setPrompt(false);
    onSaved();
  };

  return (
    <form className="capture__token" onSubmit={submit} data-testid="write-token-form">
      <GlassField
        label="Write token"
        htmlFor={id}
        hint="This API requires a token for writes. It is kept in this browser (localStorage), which any script on this origin can read — fine for a single-user prototype, not for real accounts."
      >
        <GlassInput
          id={id}
          type="password"
          value={value}
          autoComplete="off"
          placeholder="API_WRITE_TOKEN"
          onChange={(event) => setValue(event.target.value)}
          data-testid="write-token-input"
        />
      </GlassField>
      <GlassButton
        size="sm"
        type="submit"
        leadingIcon={<KeyRound size={13} aria-hidden="true" />}
        data-testid="write-token-save"
      >
        Save and retry
      </GlassButton>
    </form>
  );
}
