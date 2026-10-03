"""Attaching sidecars to an asset: the request a workflow sends, and what it gets back.

Shaped for `curl` from a GitHub workflow that already holds the R2 credentials: upload
the files to `stagingPrefix` in the private bucket, laid out exactly as they should sit
beside `tileset.json`, then POST this with the write token. Every workflow does it through
tools/captures/attach_sidecars.py; docs/DEPLOYMENT.md ("Sidecars: one publisher") has the
whole recipe, and docs/SCENE_OBJECTS.md section 8 what each workflow sends.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.asset import AssetRead
from app.schemas.base import CamelModel
from app.services.sidecars import MAX_SIDECAR_FILES


class SidecarAttach(CamelModel):
    """Attach the files staged under `stagingPrefix` to an asset's tileset."""

    staging_prefix: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "Where the files were staged in the private bucket: "
            "`staging/assets/<asset id>/<token>/`, the token a workflow run id or similar. "
            "Every object under it is attached at its path relative to it, beside "
            "`tileset.json`."
        ),
    )
    based_on: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "The tileset URL the sidecars were computed against. The attach goes ahead "
            "when the asset's current tiles are those tiles -- the same URL, or a later "
            "generation cut by another attach, which copies the tiles unchanged -- and is "
            "refused with 409 when a republish has replaced them."
        ),
    )
    files: list[str] | None = Field(
        default=None,
        max_length=MAX_SIDECAR_FILES,
        description=(
            "Optional: exactly the paths that must be staged. A partial upload is then "
            "refused instead of attached."
        ),
    )
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keys to set on the root tile's `extras`, each replaced whole -- except a list "
            "of `{uri, ...}` entries (`inferredLayers`), merged into the current list by "
            "`uri`, so send only your own entry; `null` removes a key, and with it the "
            "kind's files where none of them is staged. A `uri` a key names must be in the "
            "new generation. Setting or removing a kind's key, like staging one of its "
            "files, drops the kinds keyed by its ids (materials and telemetry by "
            "`instances`) unless they are sent too."
        ),
    )
    rig_url: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional: the asset's `renderConfig.rigUrl`, relative to the tileset, set in "
            "the same transaction as the new URL; `null` clears it. Omitted, unchanged."
        ),
    )


class SidecarAttachment(CamelModel):
    """The generation an attach cut, and the asset now pointing at it."""

    url: str
    previous_url: str
    generation: str
    #: Objects copied from the previous generation: its tiles and the sidecars it had.
    copied: int
    #: The staged paths attached.
    staged: list[str]
    #: Sidecar kinds this attach provided.
    attached: list[str]
    #: Sidecar kinds carried over from the previous generation.
    carried: list[str]
    #: Sidecar kinds the previous generation had and this one does not: keyed by the ids of
    #: a kind this attach replaced (materials and telemetry, when it replaced `instances`)
    #: and not sent with it. Each is flagged on the asset (`asset.sidecarFlags`) with why.
    dropped: list[str]
    #: Paths beside the previous `tileset.json` that the new generation does not have: the
    #: old files of a kind this attach replaced (a directory unit's stale chunks, the
    #: unstaged siblings of a staged file) and the files of every dropped kind.
    removed: list[str]
    #: The root extras keys the new `tileset.json` declares.
    extras: list[str]
    asset: AssetRead
