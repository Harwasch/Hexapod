import { useEffect, useRef } from "react";

import { useScene } from "@/cesium/SceneContext";

/**
 * Where the data credits sit: a real cell of the HUD's bottom bar.
 *
 * CesiumJS builds its credit container inside the widget and positions it absolutely,
 * which is how it ended up over the tool rail and under panels. The providers' terms need
 * it on screen, so instead of chasing it with offsets the element itself is moved into
 * this slot; the layout then gives it space like any other surface. It is moved back on
 * unmount so the viewer can tear down the tree it built.
 */
export function CreditSlot() {
  const scene = useScene();
  const slot = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const host = slot.current;
    const credits = scene?.viewer.bottomContainer as HTMLElement | undefined;
    if (!host || !credits) return;
    const home = credits.parentElement;
    host.appendChild(credits);
    return () => {
      if (home?.isConnected) home.appendChild(credits);
      else credits.remove();
    };
  }, [scene]);
  return <div ref={slot} className="credit-slot" data-testid="credits" />;
}
