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
    """How a reconstruction was placed on the globe."""

    EXIF_GPS = "exif-gps"
    ARKIT = "arkit"
    MANUAL = "manual"
    NONE = "none"


class ScaleSource(StrEnum):
    """Where metric scale came from.

    ``unresolved`` is a first-class answer, not a missing value: COLMAP alone recovers
    geometry up to scale, and a measurement taken off an unresolved reconstruction is
    meaningless rather than merely imprecise.
    """

    ARKIT = "arkit"
    EXIF_GPS = "exif-gps"
    MANUAL = "manual"
    UNRESOLVED = "unresolved"


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
