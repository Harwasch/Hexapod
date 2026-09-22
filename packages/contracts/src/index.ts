/**
 * API contract for the digital twin.
 *
 * The OpenAPI document in ./openapi.json is exported from apps/api and the
 * types in ./generated are produced from it with openapi-typescript. Never
 * hand-edit the generated file; run `pnpm contracts:generate`.
 */
import type { components, paths } from "./generated/api";

export type { components, paths };

type Schemas = components["schemas"];

export type Site = Schemas["SiteRead"];
export type SiteSummary = Schemas["SiteSummary"];
export type SiteCreate = Schemas["SiteCreate"];
export type SiteUpdate = Schemas["SiteUpdate"];
export type SiteQuality = Schemas["SiteQuality"];

export type SiteAsset = Schemas["AssetRead"];
export type AssetCreate = Schemas["AssetCreate"];
export type AssetInput = Schemas["AssetBase"];
export type AssetUpdate = Schemas["AssetUpdate"];
export type AssetSource = SiteAsset["source"];
export type CesiumIonSource = Schemas["CesiumIonSource"];
export type TilesUrlSource = Schemas["TilesUrlSource"];
export type RenderConfig = Schemas["RenderConfig"];
export type ResolutionMetadata = Schemas["ResolutionMetadata"];

export type Representation = Schemas["Representation"];
export type AssetProvider = Schemas["AssetProvider"];

export type Layer = Schemas["LayerRead"];
export type LayerCreate = Schemas["LayerCreate"];
export type LayerUpdate = Schemas["LayerUpdate"];
export type LayerCategory = Schemas["LayerCategory"];
export type LayerSourceType = Schemas["LayerSourceType"];
export type LayerSource = Layer["source"];
export type LayerRenderMetadata = Schemas["RenderMetadata"];
export type LegendMetadata = Schemas["LegendMetadata"];

export type CameraBookmark = Schemas["CameraBookmarkRead"];
export type CameraBookmarkCreate = Schemas["CameraBookmarkCreate"];

export type Attribution = Schemas["Attribution"];
export type LicenseMetadata = Schemas["LicenseMetadata"];
export type Provenance = Schemas["Provenance"];
export type TemporalExtent = Schemas["TemporalExtent"];
export type GeoPosition = Schemas["GeoPosition"];
export type BoundingBox = Schemas["BoundingBox"];
export type GeoJsonPolygon = Schemas["Polygon"];
export type GeoJsonMultiPolygon = Schemas["MultiPolygon"];
export type Footprint = GeoJsonPolygon | GeoJsonMultiPolygon;

export type IonStatus = Schemas["IonStatus"];
export type IonAssetMetadata = Schemas["IonAssetMetadata"];
export type HealthStatus = Schemas["HealthStatus"];
export type Problem = Schemas["Problem"];

export const REPRESENTATIONS = [
  "gaussian-splat",
  "mesh",
  "point-cloud",
  "terrain",
  "imagery",
] as const satisfies readonly Representation[];

export const LAYER_CATEGORIES = [
  "reality",
  "terrain",
  "imagery",
  "hydrology",
  "land-cover",
  "ecology",
  "infrastructure",
  "my-data",
] as const satisfies readonly LayerCategory[];

export const API_PREFIX = "/api/v1";

export type PlannerStatus = Schemas["PlannerStatus"];
export type PlanDraftRequest = Schemas["PlanDraftRequest"];
export type PlanDraft = Schemas["PlanDraft"];
export type PlanStep = Schemas["PlanStep"];
export type PlanCadence = PlanDraft["cadence"];
export type PlanRecord = Schemas["PlanRead"];
export type PlanRecordCreate = Schemas["PlanCreate"];
export type PlanRecordRevise = Schemas["PlanRevise"];
export type PlanRecordStatus = PlanRecord["status"];
export type PlanRevisionRecord = Schemas["PlanRevisionRead"];
export type PlanClarification = Schemas["Clarification"];
export type PlanAnswerValue = string | number | boolean;
export type GroundOutline = Schemas["Outline"];
export type OutlineRequest = Schemas["OutlineRequest"];

/** Captures: an upload session and the source files it is made of. */
export type Capture = Schemas["CaptureRead"];
export type CaptureDetail = Schemas["CaptureDetail"];
export type CaptureCreate = Schemas["CaptureCreate"];
export type CaptureKind = Schemas["CaptureKind"];
export type CaptureStatus = Schemas["CaptureStatus"];
export type CaptureFile = Schemas["CaptureFileRead"];
export type CaptureFileCreate = Schemas["CaptureFileCreate"];
export type CaptureFileComplete = Schemas["CaptureFileComplete"];
export type UploadStatus = Schemas["UploadStatus"];
/** One bounded run of presigned part URLs — never the whole upload. */
export type UploadWindow = Schemas["UploadWindow"];
export type PresignedPart = Schemas["PresignedPart"];
export type UploadedPart = Schemas["CaptureFilePart"];
export type CaptureFileUpload = Schemas["CaptureFileUpload"];
export type UploadPartsRequest = Schemas["CaptureFilePartsRequest"];

/** Jobs: one pipeline run over a capture, queued by the API and run by a worker. */
export type Job = Schemas["JobRead"];
export type JobCreate = Schemas["JobCreate"];
export type JobStep = Schemas["JobStepRead"];
export type JobArtifact = Schemas["ArtifactRead"];
export type JobRetry = Schemas["JobRetry"];
export type JobStepLog = Schemas["JobStepLog"];
export type ArtifactKind = Schemas["ArtifactKind"];

/** The data console (admin.html): outputs with their provenance, and storage reconciled. */
export type ArtifactRow = Schemas["ArtifactRow"];
export type ArtifactReference = Schemas["ArtifactReference"];
export type StorageReconciliation = Schemas["StorageReconciliation"];
export type OrphanObject = Schemas["OrphanObject"];
export type MissingObject = Schemas["MissingObject"];

/** The recipe catalogue: what a run can be, read from tools/pipeline rather than restated. */
export type PipelineCatalogue = Schemas["PipelineCatalogue"];
export type RecipeRead = Schemas["RecipeRead"];
export type RecipeStage = Schemas["RecipeStageRead"];
export type RecipeGpu = Schemas["RecipeGpu"];
export type PipelineProvider = Schemas["ProviderRead"];
export type RunStatus = Schemas["RunStatus"];
export type GeorefMethod = Schemas["GeorefMethod"];
export type ScaleSource = Schemas["ScaleSource"];
