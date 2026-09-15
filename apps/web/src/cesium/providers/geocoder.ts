import {
  IonGeocodeProviderType,
  IonGeocoderService,
  Rectangle,
  Cartographic,
  Math as CesiumMath,
  type Scene,
} from "cesium";

import type { Geocoder, GeocodeResult } from "../types";

/** Cesium ion geocoding (Bing by default; Google when the photorealistic world is active). */
export class IonGeocoder implements Geocoder {
  readonly name = "Cesium ion";
  readonly attribution: string;
  private readonly service: IonGeocoderService;

  constructor(scene: Scene, provider: "default" | "google" = "default") {
    this.service = new IonGeocoderService({
      scene,
      geocodeProviderType:
        provider === "google" ? IonGeocodeProviderType.GOOGLE : IonGeocodeProviderType.DEFAULT,
    });
    this.attribution =
      provider === "google" ? "Geocoding by Google via Cesium ion" : "Geocoding by Cesium ion";
  }

  async search(query: string): Promise<GeocodeResult[]> {
    const results = await this.service.geocode(query);
    return results.map((result) => {
      const destination = result.destination;
      if (destination instanceof Rectangle) {
        return {
          label: result.displayName,
          destination: {
            kind: "rectangle",
            west: CesiumMath.toDegrees(destination.west),
            south: CesiumMath.toDegrees(destination.south),
            east: CesiumMath.toDegrees(destination.east),
            north: CesiumMath.toDegrees(destination.north),
          },
        };
      }
      const carto = Cartographic.fromCartesian(destination);
      return {
        label: result.displayName,
        destination: {
          kind: "point",
          longitude: CesiumMath.toDegrees(carto.longitude),
          latitude: CesiumMath.toDegrees(carto.latitude),
        },
      };
    });
  }
}

interface NominatimResult {
  display_name: string;
  boundingbox: [string, string, string, string];
  lat: string;
  lon: string;
}

/** OpenStreetMap Nominatim fallback (low volume, attribution shown). */
export class NominatimGeocoder implements Geocoder {
  readonly name = "Nominatim";
  readonly attribution = "Search results © OpenStreetMap contributors (Nominatim)";

  async search(query: string, signal?: AbortSignal): Promise<GeocodeResult[]> {
    const url = new URL("https://nominatim.openstreetmap.org/search");
    url.searchParams.set("format", "jsonv2");
    url.searchParams.set("limit", "6");
    url.searchParams.set("q", query);
    const response = await fetch(url, { signal, headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Nominatim returned HTTP ${response.status}`);
    const items = (await response.json()) as NominatimResult[];
    return items.map((item) => {
      const [south, north, west, east] = item.boundingbox.map(Number) as [
        number,
        number,
        number,
        number,
      ];
      return {
        label: item.display_name,
        destination: { kind: "rectangle", west, south, east, north },
        attribution: this.attribution,
      };
    });
  }
}

/** Tries ion first and falls back to Nominatim when the ion token lacks the geocode scope. */
export class FallbackGeocoder implements Geocoder {
  readonly name = "Search";
  readonly attribution: string;
  private usePrimary = true;

  constructor(
    private readonly primary: Geocoder,
    private readonly secondary: Geocoder,
  ) {
    this.attribution = primary.attribution;
  }

  async search(query: string, signal?: AbortSignal): Promise<GeocodeResult[]> {
    if (this.usePrimary) {
      try {
        return await this.primary.search(query, signal);
      } catch (error) {
        if (signal?.aborted) throw error;
        console.warn("geocoder: primary failed, falling back", error);
        this.usePrimary = false;
      }
    }
    return this.secondary.search(query, signal);
  }
}
