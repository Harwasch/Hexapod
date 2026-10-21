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
export type ScaleEvidence = Schemas["ScaleEvidence"];
export type ScaleEvidenceInput = Schemas["ScaleEvidenceInput"];
export type AssetScaleUpdate = Schemas["AssetScaleUpdate"];
export type ResolutionMetadata = Schemas["ResolutionMetadata"];

export type Representation = Schemas["Representation"];
export type AssetProvider = Schemas["AssetProvider"];
export type AssetSourceKind = Schemas["AssetSourceKind"];

export type Epoch = Schemas["EpochRead"];
export type SiteFrame = Schemas["SiteFrameRead"];
/** An epoch's confidence map, summarised (ADR 0016); null on an epoch is "not measured". */
export type EpochConfidence = Schemas["EpochConfidenceRead"];
export type AlignmentCap = Schemas["AlignmentCapRead"];
export type WeakAreasRequest = Schemas["WeakAreasRequest"];
export type WeakAreas = Schemas["WeakAreasRead"];
export type WeakArea = Schemas["WeakArea"];
/** The cells a re-scan left weak (`rescan-gate-v1`): what an inferred fill may fill. */
export type FillTargets = Schemas["FillTargetsRead"];
export type CheckTarget = Schemas["CheckTargetRead"];
export type CheckTargetRole = Schemas["CheckTargetRole"];
export type EpochStatus = Schemas["EpochStatus"];
export type EpochJoinMethod = Schemas["EpochJoinMethod"];
export type EpochPlatform = Schemas["EpochPlatform"];
export type ObservedAtSource = Schemas["ObservedAtSource"];
export type HeightDatum = Schemas["HeightDatum"];

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
export type LandArea = Schemas["LandRead"];
export type LandCreate = Schemas["LandCreate"];
export type LandRevise = Schemas["LandRevise"];
export type BoundarySource = Schemas["BoundarySource"];
export type BoundaryRevision = Schemas["BoundaryRevisionRead"];
export type BoundaryResult = Schemas["BoundaryResult"];
export type OutlineRequest = Schemas["OutlineRequest"];

/** Captures: an upload session and the source files it is made of. */
export type Capture = Schemas["CaptureRead"];
export type CaptureDetail = Schemas["CaptureDetail"];
/** A capture a phone started with the phone key, and its upload token. */
export type PhoneCapture = Schemas["PhoneCapture"];
export type PhoneRefine = Schemas["PhoneRefine"];
/** The quality bar's verdict on a capture's latest finished run (null before one). */
export type CaptureQuality = Schemas["CaptureQuality"];
export type QualityRoi = Schemas["QualityRoi"];
export type QualityTip = Schemas["QualityTip"];
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

/** A run while it runs, for the live viewer: cameras as solved, splats as they train. */
export type LiveState = Schemas["LiveState"];
export type LiveCameras = Schemas["LiveCameras"];
export type LiveSplat = Schemas["LiveSplat"];
export type LiveStage = Schemas["LiveStage"];

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

/** Private land exploration and persistent research. */
export type Investigation = Schemas["InvestigationRead"];
export type InvestigationDetail = Schemas["InvestigationDetail"];
export type ResearchRun = Schemas["RunRead"];
export type LandEvidence = Schemas["EvidenceRead"];
export type LandFinding = Schemas["FindingRead"];
export type ResearchArtifact = Schemas["ResearchArtifactRead"];
export type ResearchEvent = Schemas["EventRead"];

export type LandCandidate = Schemas["LandCandidate"];
export type CandidateResult = Schemas["CandidateResult"];
export type SelectionInterpretation = Schemas["SelectionInterpretation"];
export type LandMapGeometry = LandCandidate["geometry"];

/** Research worker presence and point facts ("What's here"): stream research-status. */
export type ResearchStatus = Schemas["ResearchStatus"];
export type ResearchWorkerStatus = Schemas["ResearchWorkerStatus"];
export type ResearchQueueStatus = Schemas["ResearchQueueStatus"];
export type PlaceFact = Schemas["PlaceFact"];
export type PlaceFactKind = PlaceFact["kind"];
/** Land as mission ground (stream loop): a saved Land's zones, outline-sized estimates, and
 * plans held in a workspace. */
export type LandGround = Schemas["LandGround"];
export type LandGroundItem = Schemas["LandGroundItem"];
export type ZoneCoverage = Schemas["ZoneCoverage"];
export type MachineClassId = ZoneCoverage["machineClass"];
export type WorkspaceMissionCreate = Schemas["WorkspaceMissionCreate"];
/** Land-scale data (stream showdata): seasonal Sentinel-2 signatures, USGS 3DEP bare-earth
 * rasters and USGS NHD flowlines around a saved Land. */
export type RasterDataset = NonNullable<Schemas["RasterRequest"]["dataset"]>;
export type SignatureInfo = Schemas["SignatureInfo"];
export type SignatureFeature = Schemas["SignatureFeature"];
export type LandHydrography = Schemas["LandHydrography"];
export type HydrographyFeature = Schemas["HydrographyFeature"];
/** The Fog of Knowing (stream fog): a lon/lat grid of how fine and fresh the stored knowledge of
 * a Land is, the records behind a cell, and the foggiest patches to scout. */
export type LonLatGrid = Schemas["LonLatGrid"];
export type KnowledgeGrid = Schemas["KnowledgeGridRead"];
export type KnowledgeRecord = Schemas["KnowledgeRecordRead"];
export type KnowledgeCell = Schemas["KnowledgeCellRead"];
export type KnowledgeCellRecord = Schemas["KnowledgeCellRecord"];
export type KnowledgePolicy = Schemas["KnowledgePolicyRead"];
export type KnowledgeClassId = KnowledgeRecord["classId"];
export type KnowledgeNotCounted = Schemas["NotCountedRead"];
export type KnowledgeScoutRequest = Schemas["KnowledgeScoutRequest"];
export type KnowledgeScoutRead = Schemas["KnowledgeScoutRead"];
export type KnowledgeScoutArea = Schemas["KnowledgeScoutArea"];
export type ScoutCapture = Schemas["ScoutCapture"];
export type ScoutRescanRef = Schemas["ScoutRescanRef"];
/** Dated public imagery for the Time Lantern (stream lantern): NAIP years, dry-season
 * Sentinel-2 scenes and USGS historical map editions, each with its date's meaning, licence
 * and stated positional accuracy. */
export type DatedImageryCatalog = Schemas["DatedImageryCatalog"];
export type DatedImagerySource = Schemas["DatedImagerySource"];
export type DatedImageryPart = Schemas["DatedImageryPart"];
export type DatedImageryProviderStatus = Schemas["DatedImageryProviderStatus"];
export type StatedAccuracy = Schemas["StatedAccuracy"];
/** Show once, everywhere (stream showonce): how alike every 10 m cell of a Land is to one tapped
 * cell by seasonal Sentinel-2 signature, and the alike cells as areas for a plan. */
export type SignatureSimilarity = Schemas["SignatureSimilarityRead"];
export type SignatureSeed = Schemas["SignatureSeed"];
export type SignatureScene = Schemas["SignatureScene"];
export type SignatureZonesRequest = Schemas["SignatureZonesRequest"];
export type SignatureZones = Schemas["SignatureZonesRead"];
export type SignatureZone = Schemas["SignatureZone"];
// Workable ground: an outline less what an assumed machine class cannot work there.
export type WorkableGroundRequest = Schemas["WorkableGroundRequest"];
export type WorkableGround = Schemas["WorkableGroundRead"];
export type WorkableLeftOut = Schemas["WorkableLeftOut"];

/** A search query embedded with the model scan objects were described with (search by meaning). */
export type TextEmbedding = Schemas["TextEmbedding"];
