"""Many small object-storage requests at once, failing as soon as one fails.

A run's footprint in the bucket is hundreds of small objects -- a packaged tileset is a
`tileset.json` and one `.glb` per tile (514 for the first real Lane 2 capture), a frames
artifact ~100 JPEGs -- and every one of them is a request whose time is mostly a round
trip to R2, not bytes. One at a time, the 514-tile publish took about eight minutes on
the worker. The same requests eight at a time spend the round trips side by side.

Shared by the three places that move a directory of objects: the cloud transfer
(`cloud.ObjectStoreTransfer`), the artifact upload (`outputs.upload_artifact`) and the
publish copy (`publish.Publisher.publish_tree`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

#: How many objects move at once. Eight keeps a directory's requests within botocore's
#: connection pool (`S3Storage` sizes it for eight of these, each of which may itself be a
#: multipart transfer), and for uploads read whole -- a stage's frames -- within 8 x 2 MB
#: of the worker's memory. Past eight the round trips are no longer what dominates.
TRANSFER_WORKERS = 8


def each[T, R](
    work: Callable[[T], R], items: Sequence[T], *, workers: int = TRANSFER_WORKERS
) -> list[R]:
    """`work` over `items`, `workers` at a time; the results in the order of `items`.

    The first failure is raised once the requests already in flight have finished, and
    nothing that had not started yet is started: a publish that has failed must not go on
    copying a tileset that will not be registered, and a caller that cleans up after a
    failure must not race requests still being issued.
    """
    if len(items) <= 1 or workers <= 1:
        return [work(item) for item in items]
    pool = ThreadPoolExecutor(max_workers=min(workers, len(items)))
    try:
        futures = [pool.submit(work, item) for item in items]
        wait(futures, return_when=FIRST_EXCEPTION)
        for future in futures:
            error = future.exception() if future.done() and not future.cancelled() else None
            if error is not None:
                raise error
        return [future.result() for future in futures]
    finally:
        # On a failure: drop what has not started, and wait for what has.
        pool.shutdown(wait=True, cancel_futures=True)
