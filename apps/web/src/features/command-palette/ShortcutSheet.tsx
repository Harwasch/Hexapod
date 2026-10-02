import { GlassSheet, Kbd } from "@twin/ui";

import { env } from "@/app/env";
import { hotkeyKeys, hotkeySheet } from "@/app/hotkeys";
import { useUi } from "@/state/ui";

/** `?`: every key the console answers to, printed from the one registry (`app/hotkeys.ts`). */
export function ShortcutSheet() {
  const open = useUi((s) => s.shortcutsOpen);
  const setOpen = useUi((s) => s.setShortcutsOpen);
  return (
    <GlassSheet
      open={open}
      onOpenChange={setOpen}
      title="Keyboard shortcuts"
      description="Keys work from the map, not while typing. The command box runs every action by name."
      side="center"
      className="shortcuts"
      testId="shortcut-sheet"
    >
      <div className="shortcuts__grid">
        {hotkeySheet(env.devToolsEnabled).map(({ group, hotkeys }) => (
          <section
            key={group}
            className="shortcuts__group"
            aria-labelledby={`shortcuts-${group.replace(/\s+/g, "-")}`}
          >
            <h3 className="glass-eyebrow" id={`shortcuts-${group.replace(/\s+/g, "-")}`}>
              {group}
            </h3>
            <dl className="shortcuts__list">
              {hotkeys.map((hotkey) => (
                <div key={hotkey.label} className="shortcuts__row">
                  <dt>{hotkey.label}</dt>
                  <dd>
                    {hotkeyKeys(hotkey).map((key) => (
                      <Kbd key={key}>{key}</Kbd>
                    ))}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </GlassSheet>
  );
}
