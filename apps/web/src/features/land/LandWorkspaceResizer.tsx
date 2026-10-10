/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex -- WAI-ARIA window splitter is focusable and supports arrow keys. */
export function LandWorkspaceResizer({
  panelWidth,
  setPanelWidth,
}: {
  panelWidth: number;
  setPanelWidth: (value: number | ((previous: number) => number)) => void;
}) {
  return (
    <div
      role="separator"
      aria-label="Resize land workspace"
      aria-orientation="vertical"
      aria-valuemin={340}
      aria-valuemax={720}
      aria-valuenow={panelWidth}
      tabIndex={0}
      className="land-resizer"
      onPointerDown={(event) => event.currentTarget.setPointerCapture(event.pointerId)}
      onPointerMove={(event) => {
        if (!event.currentTarget.hasPointerCapture(event.pointerId)) return;
        const left = event.currentTarget.closest(".land-panel")?.getBoundingClientRect().left ?? 90;
        setPanelWidth(Math.max(340, Math.min(720, window.innerWidth - 440, event.clientX - left)));
      }}
      onKeyDown={(event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        setPanelWidth((width) =>
          Math.max(
            340,
            Math.min(720, window.innerWidth - 440, width + (event.key === "ArrowRight" ? 24 : -24)),
          ),
        );
      }}
    />
  );
}
