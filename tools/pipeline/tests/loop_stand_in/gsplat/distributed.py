"""gsplat v1.5.3's `cli`, as it behaves on one GPU: call `fn(0, 0, 1, args)` in-process.

The real one (`gsplat/distributed.py` at the tag) asserts CUDA, counts devices, and with
one device calls `_distributed_worker(0, 1, fn, args, None, verbose)`, which calls
`fn(local_rank, world_rank, world_size, args)` -- this, without the CUDA check. It lives
beside the stand-in trainer because `converge_trainer.py` puts the trainer's directory
first on `sys.path`, as `python simple_trainer.py` does; the real `examples/` directory
has no `gsplat` package in it, so there the real one is found.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def cli(fn: Callable[..., Any], args: Any, verbose: bool = False) -> bool:
    fn(0, 0, 1, args)
    return True
