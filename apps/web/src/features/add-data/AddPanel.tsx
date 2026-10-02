import { Link2, UploadCloud } from "lucide-react";

import { GlassSegmentedControl } from "@twin/ui";

import { useUi, type AddTab } from "@/state/ui";

import { CaptureUploads, ClearFinishedUploads } from "../captures/CaptureUploads";
import { FloatingPanel } from "../shell/FloatingPanel";
import { LinkSource } from "./LinkSource";

const TABS: { value: AddTab; label: string; icon: typeof UploadCloud }[] = [
  { value: "upload", label: "Upload a capture", icon: UploadCloud },
  { value: "link", label: "Link a source", icon: Link2 },
];

/**
 * Add: everything that puts something new on the globe, in one panel with two tabs.
 *
 * - **Upload a capture** — drop a video, photos or a splat, or send them from a phone; the
 *   list below follows each one through upload and processing.
 * - **Link a source** — register a site or a layer that is already hosted (ion, 3D Tiles,
 *   GeoJSON, imagery, STAC).
 *
 * Upload comes first because that is what most people arriving here have: files, not URLs.
 * A panel in the left dock, not a sheet over the map: an upload runs for minutes and the
 * camera stays usable the whole time.
 */
export function AddPanel() {
  const open = useUi((s) => s.activePanel === "add");
  const setPanel = useUi((s) => s.setPanel);
  const tab = useUi((s) => s.addTab);
  const setTab = useUi((s) => s.setAddTab);
  return (
    <FloatingPanel
      open={open}
      title="Add"
      wide
      onClose={() => setPanel(null)}
      testId="add-panel"
      actions={tab === "upload" && <ClearFinishedUploads />}
    >
      <div className="glass-stack">
        <GlassSegmentedControl
          aria-label="What to add"
          data-testid="add-mode"
          block
          value={tab}
          onValueChange={setTab}
          options={TABS.map(({ value, label, icon: Icon }) => ({
            value,
            label,
            icon: <Icon size={14} aria-hidden="true" />,
          }))}
        />
        {tab === "upload" ? (
          <CaptureUploads open={open} />
        ) : (
          <LinkSource onDone={() => setPanel(null)} />
        )}
      </div>
    </FloatingPanel>
  );
}
