/** Native fullscreen where available; the same panel fills the page otherwise. */
export function liveFullscreen(
  panel: HTMLElement,
  button: HTMLElement,
  root: HTMLElement,
): () => void {
  const abort = new AbortController();
  const options = { signal: abort.signal };
  let expanded = false;
  let stopped = false;
  const update = (value: boolean): void => {
    expanded = value;
    panel.classList.toggle("lc-expanded", value);
    root.classList.toggle("lc-fullscreen", value);
    button.setAttribute("aria-label", value ? "Exit fullscreen" : "Enter fullscreen");
    button.setAttribute("aria-pressed", String(value));
  };
  const leave = async (): Promise<void> => {
    if (document.fullscreenElement === panel)
      await document.exitFullscreen().catch(() => undefined);
    update(false);
  };
  button.addEventListener(
    "click",
    () => {
      if (expanded) {
        void leave();
        return;
      }
      update(true);
      if (panel.requestFullscreen)
        void panel
          .requestFullscreen()
          .then(() => {
            if (stopped) void leave();
          })
          .catch(() => {
            /* Fixed inset fallback also covers denied native fullscreen. */
          });
    },
    options,
  );
  document.addEventListener(
    "fullscreenchange",
    () => update(document.fullscreenElement === panel),
    options,
  );
  document.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Escape" && expanded) {
        event.preventDefault();
        void leave();
      }
    },
    options,
  );
  return () => {
    stopped = true;
    abort.abort();
    void leave();
  };
}
