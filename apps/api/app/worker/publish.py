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

So: two buckets, one key scheme. A published object keeps the key it already had, which
means nothing has to invent a second naming convention and a key in a URL can be read
straight back as the artifact row it came from. The private bucket keeps everything; the
public one holds only what `register` decided to put on the globe.

**With no public bucket configured this is a no-op**, and the URL a site gets is the one
it would have got before. That is how a fresh checkout, the test suite and a MinIO dev
loop keep working with one bucket and no ceremony. It is also why `create_app` refuses to
start a *production* deployment that has storage and a public URL but no separate public
bucket: the convenient case and the dangerous case look identical from inside the process,
so the deployment that must not be convenient is the one that gets checked. See
`app/main.py`.

**What a browser is told about caching travels with the object.** `outputs.upload_artifact`
writes every run output with its `Cache-Control` (`outputs.cache_control_for`: a year and
immutable for a run's binary files, five minutes for JSON), and a copy made here keeps the
metadata it was written with -- so the published tiles carry the same lifetime whether a
browser reads them from the bucket's public URL or through the tile proxy
(`functions/r2/[[path]].js`), which applies the same rule. An object uploaded before that
was set has none, and is published as it is.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.storage import ObjectStorage
from app.storage.null import StorageUnavailableError
from app.worker.parallel import TRANSFER_WORKERS, each

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
    single-bucket behaviour this replaced.
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

    def publish_object(self, key: str) -> str | None:
        """Copy one object across and return its public URL."""
        if not self.splits_buckets:
            return self.url(key)
        try:
            self.public.copy_object(self.private.bucket, key, key)
        except StorageUnavailableError:
            return None
        except Exception as error:
            raise PublishError(f"could not publish {key}: {error}") from error
        return self.url(key)

    def publish_tree(self, prefix: str, entry: str) -> str | None:
        """Copy every object under `prefix` across, and return `entry`'s public URL.

        A 3D tileset is a `tileset.json` and the tiles it names, so publishing the entry
        file alone would produce a site whose manifest resolves to nothing. `prefix` is
        the directory; `entry` is the key inside it that the viewer is pointed at.

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
        public bucket are unreachable from anything a viewer is given.
        """
        if not self.splits_buckets:
            return self.url(entry)
        keys = self._keys_under(prefix)
        if not keys:
            raise PublishError(f"nothing to publish under {prefix}")
        if entry not in keys:
            raise PublishError(
                f"{prefix} holds {len(keys)} objects, but not {entry}; publishing none of them"
            )
        members = [key for key in keys if key != entry]
        try:
            each(self._copy, members, workers=PUBLISH_WORKERS)
        except StorageUnavailableError:
            return None
        except Exception as error:
            raise PublishError(f"could not publish {prefix}: {error}") from error
        # The barrier: only now that every member is in place does the root become public.
        return self.publish_object(entry)

    def _copy(self, key: str) -> None:
        """One object across, keeping its metadata (content type, Cache-Control)."""
        self.public.copy_object(self.private.bucket, key, key)

    def _keys_under(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token: str | None = None
        while True:
            page = self.private.list_objects(prefix, continuation_token=token)
            keys.extend(item.key for item in page.objects)
            if len(keys) > MAX_PUBLISHED_OBJECTS:
                raise PublishError(
                    f"{prefix} holds more than {MAX_PUBLISHED_OBJECTS} objects; refusing to "
                    "publish a fraction of a tileset"
                )
            token = page.next_continuation_token
            if token is None:
                return keys
