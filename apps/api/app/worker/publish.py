"""Copying the few objects a browser must fetch into the bucket the world can read.

Everything a run produces lands in one private bucket: the raw upload under `captures/`,
and under `runs/<job>/<stage>/` the frames, the poses, the checkpoints, the logs and the
packaged tileset. Exactly two of those are things a browser goes and gets -- the tileset
and the thumbnail -- and until this module existed, making them reachable meant making
the bucket reachable.

That is not a figure of speech. Cloudflare's public-bucket feature "allows users to
expose the contents of their R2 buckets directly to the Internet", with no way to scope
it to a prefix, and a custom domain behaves the same way. One bucket plus a public URL
therefore publishes every scan anyone has ever uploaded, along with every log line and
every intermediate artifact. The keys carry UUIDs so they are not enumerable *through the
bucket*, but they are not secret: they are in the database, in the Outputs view, and in
the URLs the console renders.

So: two buckets, one key scheme. A published object keeps the key it already had, with
one segment added -- the publish's *generation*, `runs/<job>/p<generation>/<stage>/...`
(`app/services/published.py`) -- so a key in a URL can still be read straight back as the
artifact row it came from. The private bucket keeps everything; the public one holds only
what `register` decided to put on the globe.

**Every publish writes keys nobody has seen.** A run's own keys are rewritten -- a phone's
Refine re-runs the same job from `train`, and `package` writes the same tile names with
new geometry -- and publishing used to copy them across under those names while they were
served `immutable` for a year. A Refine then mixed the preview's cached tiles with the new
`tileset.json`, and a republish that failed half way had already overwritten part of the
live site. Now each publish copies into a generation of its own and the site moves to it
once it is complete; the live generation is never written again, which is what makes
`immutable` true. The generation is a hash of what is published (`Publisher.generation`),
so the same bytes published twice land on the same keys.

**With no public bucket configured this is a no-op**, and the URL a site gets is the one
it would have got before. That is how a fresh checkout, the test suite and a MinIO dev
loop keep working with one bucket and no ceremony. It is also why `create_app` refuses to
start a *production* deployment that has storage and a public URL but no separate public
bucket: the convenient case and the dangerous case look identical from inside the process,
so the deployment that must not be convenient is the one that gets checked. See
`app/main.py`.

**What a browser is told about caching travels with the object.** Every copy made here
states its metadata: the `Cache-Control` of the key it is copied *to*
(`outputs.cache_control_for`: a year and immutable inside a generation, five minutes for
JSON) and the content type the original was uploaded with (`outputs.member_content_type`
for a tileset's members, read off the object for a single file). So the published tiles
carry the same lifetime whether a browser reads them from the bucket's public URL or
through the tile proxy (`functions/r2/[[path]].js`), which applies the same rule; the run's
own keys, which a Refine rewrites, are uploaded with the short one.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections.abc import Iterable
from dataclasses import dataclass

from app.services.published import GENERATION_LENGTH, published_key
from app.storage import ObjectStorage
from app.storage.base import ObjectSummary
from app.storage.null import StorageUnavailableError
from app.worker.outputs import cache_control_for, member_content_type
from app.worker.parallel import TRANSFER_WORKERS, each

log = logging.getLogger("app.worker")

#: A tileset is many files and `list_objects` pages. This is the ceiling on one publish,
#: and it is a refusal rather than a truncation: half a tileset in the public bucket is a
#: site that renders a hole, which is worse than a site that says it could not publish.
MAX_PUBLISHED_OBJECTS = 20_000

#: Copies in flight at once in one publish. A `CopyObject` moves no bytes through the
#: worker, so what eight at a time saves is round trips -- the whole cost of a publish.
PUBLISH_WORKERS = TRANSFER_WORKERS


class PublishError(RuntimeError):
    """A publish that could not be completed. The run still succeeded; the site is not on
    the globe, and the caller says so rather than pointing a viewer at a partial copy."""


@dataclass(frozen=True)
class Publisher:
    """Where published objects go, and where a browser then finds them.

    `public` is the bucket the world can read. When it is unavailable -- unconfigured,
    or a `NullStorage` -- every method falls back to `private.public_url`, which is the
    single-bucket behaviour this replaced: nothing is copied, there is no generation, and
    a site points at the run's own keys.
    """

    private: ObjectStorage
    public: ObjectStorage

    @property
    def splits_buckets(self) -> bool:
        """True where publishing actually moves an object out of the private bucket."""
        return self.public.available and self.public.bucket != self.private.bucket

    def url(self, key: str) -> str | None:
        """The public URL for a key already published, or None where there is no storage."""
        source = self.public if self.splits_buckets else self.private
        try:
            return source.public_url(key)
        except StorageUnavailableError:
            return None

    def generation(self, prefixes: Iterable[str] = (), keys: Iterable[str] = ()) -> str | None:
        """The generation one publish of these objects writes into, or None with one bucket.

        A hash of every object's key, size and ETag: the same bytes give the same
        generation, so publishing them again overwrites identical objects with identical
        ones and a browser's cached copies stay good, while any change -- a Refine's new
        tiles -- gives keys nobody has fetched. Where the store cannot vouch for the bytes
        (an object without an ETag, a listing that fails) the generation is random, which
        is always correct and only costs the reuse. Every object a publish copies should
        be named here, so that one generation holds the tileset and what goes with it.
        """
        if not self.splits_buckets:
            return None
        lines: list[str] = []
        try:
            for prefix in prefixes:
                lines += [f"{o.key}\t{o.size}\t{o.etag}" for o in self._listing(prefix)]
            for key in keys:
                head = self.private.head_object(key)
                if head is not None:
                    lines.append(f"{key}\t{head.size}\t{head.etag or ''}")
        except Exception:
            log.warning("publish: could not read what is being published; a fresh generation")
            return _fresh_generation()
        if not lines or any(line.endswith("\t") for line in lines):
            return _fresh_generation()
        digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
        return digest[:GENERATION_LENGTH]

    def publish_object(self, key: str, *, generation: str | None = None) -> str | None:
        """Copy one object into `generation` and return its public URL.

        With no generation, one of its own (from this object's bytes). The copy is
        labelled with the type the object was stored with and the lifetime of the key it
        lands on.
        """
        if not self.splits_buckets:
            return self.url(key)
        try:
            source = self.private.head_object(key)
            if source is None:
                raise PublishError(f"could not publish {key}: there is no such object")
            target = published_key(key, generation or self.generation(keys=[key]) or "")
            self.public.copy_object(
                self.private.bucket,
                key,
                target,
                content_type=source.content_type,
                cache_control=cache_control_for(target),
            )
        except StorageUnavailableError:
            return None
        except PublishError:
            raise
        except Exception as error:
            raise PublishError(f"could not publish {key}: {error}") from error
        return self.url(target)

    def publish_tree(self, prefix: str, entry: str, *, generation: str | None = None) -> str | None:
        """Copy every object under `prefix` into `generation`, and return `entry`'s URL.

        A 3D tileset is a `tileset.json` and the tiles it names, so publishing the entry
        file alone would produce a site whose manifest resolves to nothing. `prefix` is
        the directory; `entry` is the key inside it that the viewer is pointed at. With no
        generation, one of its own (from the directory's bytes).

        The entry must be among the objects listed, and is checked before anything is
        copied: a tileset whose root file is missing is a broken site that looks like a
        working one until someone opens it, and is refused without copying its tiles.

        The members are copied `PUBLISH_WORKERS` at a time, and the entry **last**, once
        every member's copy has returned. It used to be one copy and one HEAD at a time in
        listing order -- 514 tiles took ~8 minutes on the worker, and `tileset.json` was
        public while `collision.bin` and `viewcones.bin`, which sort after it, were not
        yet. With the entry last, a public `tileset.json` means everything it names is
        already there; and when any member fails, the entry is never copied, the publish
        raises, and `publish_outputs` registers no site -- the stray tiles left in the
        public bucket are in a generation nothing points at, and the live one, if there
        is one, was never touched.
        """
        if not self.splits_buckets:
            return self.url(entry)
        try:
            keys = [item.key for item in self._listing(prefix)]
        except (StorageUnavailableError, PublishError):
            raise
        except Exception as error:
            # A listing that fails is a publish that failed, and says so like any other
            # rather than escaping to the run that asked for it.
            raise PublishError(f"could not list {prefix}: {error}") from error
        if not keys:
            raise PublishError(f"nothing to publish under {prefix}")
        if entry not in keys:
            raise PublishError(
                f"{prefix} holds {len(keys)} objects, but not {entry}; publishing none of them"
            )
        generation = generation or self.generation(prefixes=[prefix]) or ""
        members = [key for key in keys if key != entry]
        try:
            each(lambda key: self._copy(key, generation), members, workers=PUBLISH_WORKERS)
            # The barrier: only now that every member is in place does the root go across.
            self._copy(entry, generation)
        except StorageUnavailableError:
            return None
        except Exception as error:
            raise PublishError(f"could not publish {prefix}: {error}") from error
        return self.url(published_key(entry, generation))

    def _copy(self, key: str, generation: str) -> None:
        """One member across, labelled as it was uploaded and with its key's lifetime."""
        target = published_key(key, generation)
        self.public.copy_object(
            self.private.bucket,
            key,
            target,
            content_type=member_content_type(key),
            cache_control=cache_control_for(target),
        )

    def _listing(self, prefix: str) -> list[ObjectSummary]:
        """Every object under the directory `prefix` (not merely sharing its first
        characters: `splat/` is not `splat_old/`), paged to the end."""
        directory = prefix.rstrip("/") + "/"
        found: list[ObjectSummary] = []
        token: str | None = None
        while True:
            page = self.private.list_objects(directory, continuation_token=token)
            found.extend(page.objects)
            if len(found) > MAX_PUBLISHED_OBJECTS:
                raise PublishError(
                    f"{prefix} holds more than {MAX_PUBLISHED_OBJECTS} objects; refusing to "
                    "publish a fraction of a tileset"
                )
            token = page.next_continuation_token
            if token is None:
                return found


def _fresh_generation() -> str:
    return secrets.token_hex(GENERATION_LENGTH // 2)
