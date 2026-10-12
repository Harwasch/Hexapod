import type { components, LandArea, LandEvidence } from "@twin/contracts";

type Investigation = components["schemas"]["InvestigationRead"];
type Detail = components["schemas"]["InvestigationDetail"];
type Run = components["schemas"]["RunRead"];
type Document = components["schemas"]["LandDocumentRead"];
type Registration = components["schemas"]["ImageRegistrationRead"];
type Fit = components["schemas"]["ImageRegistrationResult"];
type ArchiveMedia = components["schemas"]["ArchiveMedia"];

export const LAND = "11111111-1111-4111-8111-111111111111";

export const land: LandArea = {
  id: LAND,
  name: "Public fixture",
  description: "",
  revision: 2,
  areaM2: 40000,
  perimeterM: 800,
  boundary: {
    type: "Polygon",
    coordinates: [
      [
        [-77.055, 38.886],
        [-77.045, 38.886],
        [-77.045, 38.891],
        [-77.055, 38.891],
        [-77.055, 38.886],
      ],
    ],
  },
  source: {
    method: "parcel",
    label: "Washington county tax-parcel record",
    recordId: "019-210-034",
    meaning: "recorded-parcel",
    attribution: "Washington State parcels",
    url: "https://example.test/parcels",
  },
  createdAt: "2026-10-09T12:00:00Z",
  updatedAt: "2026-10-09T12:00:00Z",
};

export const photoMedia: ArchiveMedia = {
  kind: "photograph",
  title: "File:Main Street barn.jpg",
  description: "Creation date is uncertain.",
  sourceUrl: "https://commons.wikimedia.org/wiki/File:Test.jpg",
  previewUrl: "https://upload.wikimedia.org/wikipedia/commons/a/a1/Test.jpg",
  creator: "A. Photographer",
  license: "CC BY-SA 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
  sourceDate: "circa 1905–1910",
  dateMeaning: "Upload dates are not event dates.",
  location: { type: "Point", coordinates: [-77.05, 38.888] },
  locationMeaning: "catalog-coordinate",
  relevance: "The coordinate may locate the camera.",
  sourceVersion: "Commons file SHA-1 example",
};

export const photo: LandEvidence = {
  id: "photo",
  runId: "run",
  provider: "commons-place-images",
  title: "A historical view",
  attribution: "A. Photographer; Wikimedia Commons; CC BY-SA 4.0",
  excerpt: "Catalog metadata",
  license: "CC BY-SA 4.0",
  spatialRelevance: "within",
  relevanceNote: "Catalog coordinate only",
  retrievedAt: "2026-10-09T12:00:00Z",
  media: photoMedia,
};

const sheetFootprint = {
  type: "Polygon" as const,
  coordinates: [
    [
      [-77.1, 38.8],
      [-77, 38.8],
      [-77, 38.9],
      [-77.1, 38.9],
      [-77.1, 38.8],
    ],
  ],
};

export const sheetMedia: ArchiveMedia = {
  ...photoMedia,
  kind: "historical-map",
  title: "Mount Vernon, WA, 1894",
  creator: "U.S. Geological Survey",
  license: "U.S. public domain",
  licenseUrl: "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
  sourceDate: "1894",
  dateMeaning: "Catalog publication year.",
  downloadUrl: "https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/x.tif",
  location: sheetFootprint,
  locationMeaning: "catalog-footprint",
  relevance: "The sheet outline overlaps this land.",
};

export const sheet: LandEvidence = {
  ...photo,
  id: "map",
  provider: "usgs-historical-maps",
  title: "Historical sheet",
  attribution: "U.S. Geological Survey; public domain",
  license: "U.S. public domain",
  spatialRelevance: "intersects",
  media: sheetMedia,
};

export const laterSheet: LandEvidence = {
  ...sheet,
  id: "map-1956",
  media: { ...sheetMedia, title: "Mount Vernon, WA, 1956", sourceDate: "1956" },
};

export const archiveInvestigation: Investigation = {
  id: "archive-inv",
  landId: LAND,
  title: "Photographs and historical maps",
  boundaryRevision: 2,
  createdAt: "2026-10-09T12:00:00Z",
  stale: false,
};

export function run(status: Run["status"], error: string | null = null): Run {
  return {
    id: "run",
    investigationId: "archive-inv",
    kind: "archive",
    question: "",
    status,
    error,
    attempt: 1,
    budget: {},
    createdAt: "2026-10-09T12:00:00Z",
    startedAt: null,
    finishedAt: null,
  };
}

export function detail(evidence: LandEvidence[], status: Run["status"] = "succeeded"): Detail {
  return {
    investigation: archiveInvestigation,
    runs: [run(status)],
    findings: [
      {
        id: "finding",
        runId: "run",
        title: "Old photos",
        summary: "Found 1 openly licensed photograph in a bounded nearby catalog sample.",
        category: "history",
        evidenceIds: ["photo"],
        confidence: "supported",
        uncertainty: "Dates and land relevance need verification.",
        disposition: "visible",
      },
    ],
    evidence,
    artifacts: [],
    messages: [],
    page: { limit: 100, offset: 0, totals: {} },
  };
}

export const deed: Document = {
  id: "deed",
  landId: LAND,
  title: "County deed",
  filename: "deed.txt",
  mediaType: "text/plain",
  sizeBytes: 4,
  kind: "deed",
  sourceNote: "County archive",
  status: "ready",
  sha256: "a".repeat(64),
  pageCount: 2,
  extractedCharacters: 30,
  warnings: [],
  documentDate: "1998-03-02",
  parcelReferences: ["019210034"],
  createdAt: "2026-10-09T12:00:00Z",
};

export const easement: Document = {
  ...deed,
  id: "easement",
  title: "Access easement",
  kind: "easement",
  documentDate: "1971-05-01",
  parcelReferences: [],
};

export const image: components["schemas"]["ArchiveImageRead"] = {
  evidenceId: "map",
  width: 600,
  height: 400,
  sha256: "a".repeat(64),
  sourceSha256: "b".repeat(64),
  sourceUrl: "https://prd-tnm.s3.amazonaws.com/fixture",
  sourceMediaType: "image/png",
  byteSize: 100,
  sourceByteSize: 100,
  createdAt: "2026-10-09T12:00:00Z",
};

export const fit: Fit = {
  algorithm: "affine-aeqd-v1",
  crs: "local projection",
  transform: [2, 0, 0, 0, -2, 0],
  imageWidth: 600,
  imageHeight: 400,
  footprint: {
    type: "Polygon",
    coordinates: [
      [
        [-77.05, 38.88],
        [-77.04, 38.88],
        [-77.04, 38.89],
        [-77.05, 38.88],
      ],
    ],
  },
  bounds: [-77.05, 38.88, -77.04, 38.89],
  rmsErrorM: 2,
  maximumErrorM: 3,
  leaveOneOutRmsM: 5,
  controlPointCoverage: 1,
  approximateMPerPixel: 2,
  points: [],
  warnings: ["Fit error does not establish surveyed accuracy."],
};

export function registration(id: string, request: Registration["request"]): Registration {
  return {
    id,
    evidenceId: "map",
    request,
    result: fit,
    sha256: "c".repeat(64),
    byteSize: 100,
    displayWidth: 600,
    displayHeight: 400,
    createdAt: "2026-10-09T13:00:00Z",
  };
}
