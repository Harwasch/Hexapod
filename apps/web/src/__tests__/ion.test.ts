import { ArcGisMapService, Credit, CreditDisplay, Ion } from "cesium";
import { afterEach, describe, expect, it } from "vitest";

import { configureIonToken, demoteDefaultTokenNotices } from "@/cesium/ion";

/** What CesiumJS hands back from `getDefaultTokenCredit`, minus the types it does not declare. */
interface DefaultKeyedService {
  defaultAccessToken: string;
  getDefaultTokenCredit?: (providedKey: string) => Credit | undefined;
}

const ion = Ion as unknown as DefaultKeyedService;
const arcgis = ArcGisMapService as unknown as DefaultKeyedService;
const builtInIonToken = Ion.defaultAccessToken;

afterEach(() => {
  Ion.defaultAccessToken = builtInIonToken;
});

describe("the default-token setup advice CesiumJS paints over the globe", () => {
  // The fix in `cesium/ion.ts` reaches for an API the `cesium` types do not declare. If a
  // CesiumJS upgrade renames or drops it, this fails loudly instead of quietly putting a
  // paragraph of bold text back on the map.
  it("is still a screen credit CesiumJS offers up, so there is something to demote", () => {
    expect(typeof ion.getDefaultTokenCredit).toBe("function");
    expect(typeof arcgis.getDefaultTokenCredit).toBe("function");
    const credit = ion.getDefaultTokenCredit?.(builtInIonToken);
    expect(credit).toBeDefined();
    expect(credit?.html).toContain("default ion access token");
  });

  it("is moved off the screen and into the Data attribution dialog", () => {
    demoteDefaultTokenNotices();
    expect(ion.getDefaultTokenCredit?.(builtInIonToken)?.showOnScreen).toBe(false);
    expect(arcgis.getDefaultTokenCredit?.(arcgis.defaultAccessToken)?.showOnScreen).toBe(false);
  });

  it("is demoted before a custom token replaces the key it is looked up by", () => {
    expect(configureIonToken("a-real-token")).toBe("custom");
    expect(Ion.defaultAccessToken).toBe("a-real-token");
    // Looked up by the built-in token, which `configureIonToken` has just overwritten: the
    // demotion has to have happened first for this to be false rather than true.
    expect(ion.getDefaultTokenCredit?.(builtInIonToken)?.showOnScreen).toBe(false);
  });

  it("leaves the evaluation token in place, and names the state, when none is configured", () => {
    expect(configureIonToken(undefined)).toBe("default");
    expect(Ion.defaultAccessToken).toBe(builtInIonToken);
  });
});

describe("attribution", () => {
  // The regression guard for the constraint that outranks all of this: ion's terms and
  // Google's Photorealistic 3D Tiles terms both require their credit to stay visible.
  // Demoting the setup advice must not touch a provider credit's own visibility.
  it("survives the demotion: a provider's screen credit stays a screen credit", () => {
    const google = new Credit('<a href="https://about.google/brand-resource-center/">Google</a>');
    google.showOnScreen = true;
    const ionLogo = CreditDisplay.cesiumCredit;
    const ionLogoOnScreen = ionLogo.showOnScreen;
    demoteDefaultTokenNotices();
    expect(google.showOnScreen).toBe(true);
    expect(CreditDisplay.cesiumCredit.showOnScreen).toBe(ionLogoOnScreen);
    expect(CreditDisplay.cesiumCredit.html).toContain("ion-credit.png");
  });
});
