"""Domain enumerations shared by the ORM and the API schemas.

Values are stable string identifiers that appear in the OpenAPI contract.
"""

from __future__ import annotations

from enum import StrEnum


class Representation(StrEnum):
    GAUSSIAN_SPLAT = "gaussian-splat"
    MESH = "mesh"
    POINT_CLOUD = "point-cloud"
    TERRAIN = "terrain"
    IMAGERY = "imagery"


class AssetProvider(StrEnum):
    """Delivery provider for a derived, renderable asset.

    Cesium ion is the current delivery provider; it is deliberately *not* the
    canonical data model (see docs/ARCHITECTURE.md).
    """

    CESIUM_ION = "cesium-ion"
    TILES_3D_URL = "3d-tiles-url"


class LayerCategory(StrEnum):
    REALITY = "reality"
    TERRAIN = "terrain"
    IMAGERY = "imagery"
    HYDROLOGY = "hydrology"
    LAND_COVER = "land-cover"
    ECOLOGY = "ecology"
    INFRASTRUCTURE = "infrastructure"
    MY_DATA = "my-data"


class LayerSourceType(StrEnum):
    CESIUM_ION_TERRAIN = "cesium-ion-terrain"
    CESIUM_ION_IMAGERY = "cesium-ion-imagery"
    CESIUM_ION_3D_TILES = "cesium-ion-3d-tiles"
    GOOGLE_PHOTOREALISTIC = "google-photorealistic"
    TILES_3D_URL = "3d-tiles-url"
    XYZ = "xyz"
    WMTS = "wmts"
    WMS = "wms"
    ARCGIS_MAPSERVER = "arcgis-mapserver"
    GEOJSON = "geojson"
    CZML = "czml"
    MVT = "mvt"
    STAC = "stac"


class CaptureKind(StrEnum):
    """What was uploaded, which decides the lane a capture runs down.

    ``gaussian-splat`` and ``point-cloud`` deliberately repeat the identifiers
    :class:`Representation` already fixed in the contract: the same thing should not
    be spelled two ways depending on which table it is named in.
    """

    VIDEO = "video"
    IMAGES = "images"
    GAUSSIAN_SPLAT = "gaussian-splat"
    POINT_CLOUD = "point-cloud"


class CaptureStatus(StrEnum):
    """Lifecycle of an upload session.

    Deliberately the vocabulary Cesium ion already exposes through
    ``app.schemas.ion.IonAssetStatus`` (``AWAITING_FILES | NOT_STARTED | IN_PROGRESS |
    COMPLETE | ERROR``), in this module's kebab-case spelling: a capture waiting for its
    bytes, uploaded but unprocessed, processing, done or failed are the same five states
    ion means by those words. ion's ``DATA_ERROR`` is not repeated — it is a distinction
    about ion's own tiler, and here the error message carries that detail.
    """

    AWAITING_FILES = "awaiting-files"
    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"
    ERROR = "error"


class UploadStatus(StrEnum):
    """Lifecycle of one uploaded source object (an S3 multipart upload).

    Same words as :class:`CaptureStatus` where they mean the same thing; ``aborted`` is
    the one addition, because ``abort_multipart`` is a real outcome that is neither an
    error nor a completion.
    """

    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"
    ERROR = "error"
    ABORTED = "aborted"


class RunStatus(StrEnum):
    """Lifecycle of a job and of each of its steps — one vocabulary, because a step's
    states and a run's states are the same states.

    Again ion's words, plus ``cancelled``: a run stopped on purpose is not an error, and
    ion's asset lifecycle has nothing to say about it because ion has no cancel. There is
    deliberately no ``preempted`` member — a preempted step goes back to ``in-progress``
    on the next attempt, and ``job_steps.preempted_at`` and ``attempt`` record that it
    happened without inventing a state the step is never resting in.
    """

    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class GeorefMethod(StrEnum):
    """How a reconstruction was placed on the globe.

    ``visual-epoch`` (M3b): its frames were registered into an earlier epoch of the same
    site and placed by that alignment, not by its own GNSS. It is only as good as the
    epoch it was aligned to -- the uncertainty says how good -- so it is never read as a
    measurement in its own right. Declared before ``none``, where migration 0033 puts it.
    """

    EXIF_GPS = "exif-gps"
    ARKIT = "arkit"
    MANUAL = "manual"
    VISUAL_EPOCH = "visual-epoch"
    NONE = "none"


class ScaleSource(StrEnum):
    """Where metric scale came from.

    ``unresolved`` is a first-class answer, not a missing value: COLMAP alone recovers
    geometry up to scale, and a measurement taken off an unresolved reconstruction is
    meaningless rather than merely imprecise.

    ``camera-height-estimate`` is between a measurement and that: none was made, but a
    handheld phone is about 1.5 m above the ground and the reconstruction says how many of
    its units that is (tools/pipeline/scale_estimate.py). Roughly the right size, at ±20%
    or worse -- the provenance carries the figure -- and never counted as metric.

    ``visual-epoch`` (M3b): the scale of the earlier epoch it was aligned to, carried by
    the alignment's similarity. Written only when that epoch's own scale was measured; an
    epoch aligned to an estimated or unresolved one inherits that word instead
    (`worker/registration.py`). Before ``unresolved``, as 0033 adds it (and as the 0010
    test reads the type's order back).
    """

    ARKIT = "arkit"
    EXIF_GPS = "exif-gps"
    MANUAL = "manual"
    CAMERA_HEIGHT_ESTIMATE = "camera-height-estimate"
    VISUAL_EPOCH = "visual-epoch"
    UNRESOLVED = "unresolved"


class HeightDatum(StrEnum):
    """What a site frame's origin height is measured from.

    ``msl-unverified``: every frame today. GPS altitude is above mean sea level (the geoid),
    and nothing here applies a geoid model, so the number is recorded with the datum it
    probably has rather than passed off as an ellipsoid height.
    """

    ELLIPSOID_WGS84 = "ellipsoid-wgs84"
    MSL_UNVERIFIED = "msl-unverified"


class SiteFrameSource(StrEnum):
    """How a site's frame came to exist (decision D3): from its first epoch's georeference
    (``registration``), from migration 0032 (``backfill``), or re-made from the frame a run
    echoed after the site it named was deleted (``echo``)."""

    REGISTRATION = "registration"
    BACKFILL = "backfill"
    ECHO = "echo"


class EpochStatus(StrEnum):
    """An epoch's standing. Only ``published`` is written until quality gates exist (M3)."""

    PUBLISHED = "published"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class EpochJoinMethod(StrEnum):
    """How a capture found the site it is an epoch of: named it (``explicit``), was matched
    to it by location before its run (``spatial``), or made it (``new-site``)."""

    EXPLICIT = "explicit"
    SPATIAL = "spatial"
    NEW_SITE = "new-site"


class JoinPointSource(StrEnum):
    """Which point a capture was matched to a site by (``spatial`` joins): the location its
    video's container carried (``video``), else the one sent with the upload (``upload``:
    the browser's fix, or a place typed for the run). ADR 0013, "Joining a site by the
    video's GPS"."""

    VIDEO = "video"
    UPLOAD = "upload"


class EpochPlatform(StrEnum):
    """What captured an epoch, by survey tier (decision D4). Only ``phone-video`` and
    ``unknown`` are told apart until M3 reads DJI XMP and RTK."""

    RTK_DRONE = "rtk-drone"
    DRONE = "drone"
    PHONE_VIDEO = "phone-video"
    PHONE_PHOTO = "phone-photo"
    UNKNOWN = "unknown"


class ObservedAtSource(StrEnum):
    """Where an epoch's ``observed_at`` came from: the capture row's ``captured_at``, the
    run's manifest (the files' own date), or -- with neither -- the upload time
    (``created``), which is labelled as such rather than passed off as a capture date."""

    CAPTURED_AT = "captured-at"
    MANIFEST = "manifest"
    CREATED = "created"


class CheckTargetRole(StrEnum):
    """What a surveyed ground mark is for (M3b). A ``check`` target only scores an epoch's
    alignment against its survey; a ``control`` target would also constrain it, but until
    bundle adjustment takes ground control it is stored and scored exactly like a check."""

    CHECK = "check"
    CONTROL = "control"


class EpochMapKind(StrEnum):
    """A per-cell map an epoch revision carries beside its tiles (M4, ADR 0016): today its
    ``confidence`` grid; M5 adds its height ``surface``. One row per asset and kind."""

    CONFIDENCE = "confidence"


class AssetSourceKind(StrEnum):
    """Whether an asset shows what was captured (``observed``) or what was inferred to fill
    it (``inferred``, M5). One asset is one tileset of one epoch and one source kind."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class ArtifactKind(StrEnum):
    """What a stage produced. Step logs and checkpoints are not artifacts: they are
    ``job_steps.log_key`` and ``job_steps.checkpoint_key``.

    ``metadata`` was added in A7, when the worker started rowing everything it uploads.
    A stage writes JSON sidecars beside its real output — ``georef.json``,
    ``source_meta.json``, ``train_metrics.json``, ``registration.json`` — and they are
    genuinely artifacts: they are in the bucket, they are what a later stage reads, and
    an object with no row is an orphan by the console's own definition. Calling them
    ``manifest`` would have been the cheaper lie.
    """

    FRAMES = "frames"
    POSES = "poses"
    MASKS = "masks"
    SPLAT = "splat"
    DEFORMATION_FIELD = "deformation-field"
    MESH = "mesh"
    POINT_CLOUD = "point-cloud"
    TILES_3D = "3d-tiles"
    THUMBNAIL = "thumbnail"
    GROUND_SAMPLES = "ground-samples"
    MANIFEST = "manifest"
    CLIP = "clip"
    METADATA = "metadata"
