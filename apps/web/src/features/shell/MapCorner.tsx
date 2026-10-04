import { PHONE_MEDIA, useMediaQuery } from "@/lib/media";

import { NavControls } from "../nav/NavControls";
import { CreditSlot } from "./CreditSlot";

/**
 * The bottom-right pill: compass and Earth, then the data credits.
 *
 * On a phone the credits do not fit in the pill beside the status line, but Cesium ion's and
 * Google's Photorealistic 3D Tiles terms want them on screen, not behind a button. So there
 * they leave the pill for a strip of their own, a thin line right-aligned above the status
 * line (`app.css` gives it its row of the bar): every logo and the "Data attribution" link
 * stay in view, and the dialog behind that link carries the rest. The container is moved
 * from one slot to the other, never copied — only one `CreditSlot` is mounted at a time.
 */
export function MapCorner() {
  const phone = useMediaQuery(PHONE_MEDIA);
  return (
    <>
      {phone && <CreditSlot className="credit-slot--strip" />}
      <div className="glass glass--strong hud-corner" data-testid="map-corner">
        <NavControls />
        {!phone && <CreditSlot />}
      </div>
    </>
  );
}
