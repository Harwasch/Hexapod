"""One fake Modal call: `remote.execute` over a shared directory, as `infra/modal/app.py`'s
`_run` does over the bucket. Run by `modal.Function.spawn` here, as its own process.

`$FAKE_MODAL_TRANSFER` is the directory the worker's `WORKER_CLOUD_TRANSFER_DIR` names,
so the two sides move bytes through the same `LocalTransfer`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def main(directory: Path) -> int:
    pipeline = os.environ["PIPELINE_DIR"]
    if pipeline not in sys.path:
        sys.path.insert(0, pipeline)
    import remote
    from adapters import LocalTransfer
    from cloud import StageRequest

    import tests.worker_stages  # noqa: F401 - registers the test stages

    request = StageRequest.from_dict(json.loads((directory / "request.json").read_text()))
    root = directory / "sandbox"
    try:
        entered = time.time()
        outcome = remote.execute(
            request, LocalTransfer(Path(os.environ["FAKE_MODAL_TRANSFER"])), root
        ).to_dict()
        outcome["metrics"] = {
            **outcome["metrics"],
            "containerCall": 1,
            "remoteEnteredAt": round(entered, 3),
            "remoteFinishedAt": round(time.time(), 3),
        }
    except BaseException:
        (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (directory / "result.json").write_text(json.dumps(outcome), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
