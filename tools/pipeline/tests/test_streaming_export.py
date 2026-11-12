from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import host
import live
import splat_io


def test_memory_probe_counts_children_and_cgroup_peak(tmp_path: Path) -> None:
    proc, cg = tmp_path / "proc", tmp_path / "cgroup"
    cg.mkdir()
    for pid, kib in [(1, 1024), (2, 2048)]:
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "status").write_text(f"Name:\tworker\nVmRSS:\t{kib} kB\n")
    (cg / "memory.current").write_text(str(4 * 1024**2))
    (cg / "memory.peak").write_text(str(7 * 1024**2))
    (cg / "memory.max").write_text(str(8 * 1024**2))
    (cg / "memory.events").write_text("oom 1\noom_kill 1\n")
    values = host.memory_sample(proc=proc, cgroup=cg)
    assert values["hostRssMiB"] == 3
    assert values["hostMemoryPeakMiB"] == 7
    assert values["hostMemoryLimitMiB"] == 8
    assert values["hostOomKills"] == 1
    (cg / "memory.max").write_text("max")
    assert "hostMemoryLimitMiB" not in host.memory_sample(proc=proc, cgroup=cg)


def test_snapshot_reads_bounded_chunks_without_unused_sh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_live import write_splat_ply

    path = tmp_path / "in.ply"
    write_splat_ply(path, opacity=[float(i) for i in range(9)], log_scale=[0.0] * 9)
    import live_sample

    reads = []
    original = live_sample.iter_ply_rows  # type: ignore[attr-defined]

    def bounded(layout: Any, names: Any, chunk_bytes: int) -> Any:
        assert chunk_bytes <= 2 * layout.dtype.itemsize
        assert not any(n.startswith("f_rest") for n in names)
        for start, data in original(layout, names, chunk_bytes):
            reads.append(start)
            yield start, data

    monkeypatch.setattr(live_sample, "WINDOW_ROWS", 2)
    monkeypatch.setattr(live_sample, "iter_ply_rows", bounded)
    assert live.snapshot_splat(path, tmp_path / "out.spz", max_gaussians=3)[:2] == (3, 9)
    assert len(reads) >= 5


def test_streamed_trainer_export_preserves_rows_and_filters_invalids(tmp_path: Path) -> None:
    import trainer_export

    sizes = []

    def upstream(**kw: Any) -> bytes:
        sizes.append(len(kw["means"]))
        rows = kw["means"][:, 0]
        rows = rows[np.isfinite(rows)].astype("<f4")
        return (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n"
            "property float x\nend_header\n"
        ).encode() + bytes(rows.tobytes())

    means = np.array([[0.0, 1, 2], [float("nan"), 0, 0], [3.0, 1, 2], [4.0, 1, 2]])
    kw = {k: means for k in trainer_export.FIELDS}
    path = tmp_path / "trained.ply"
    trainer_export.write(upstream, {**kw, "save_to": str(path), "format": "ply"}, chunk=2)
    assert sizes == [2, 2]
    reader = splat_io.SplatReader(path)
    assert reader.count == 3
    np.testing.assert_array_equal(reader.read(0, 3)["x"], [0, 3, 4])
    assert not path.with_suffix(".ply.part").exists()
    assert path.read_bytes().split(b"end_header\n")[1] == struct.pack("<3f", 0, 3, 4)


def test_host_watch_persists_growth_and_oom_counter_delta(tmp_path: Path) -> None:
    values = iter(
        [
            {
                "hostRssMiB": 2.0,
                "hostMemoryPeakMiB": 4.0,
                "hostMemoryLimitMiB": 8.0,
                "hostOomKills": 10.0,
            },
            {
                "hostRssMiB": 7.0,
                "hostMemoryPeakMiB": 8.0,
                "hostMemoryLimitMiB": 8.0,
                "hostOomKills": 11.0,
            },
        ]
    )
    seen: list[Any] = []
    with host.HostWatch(3600, probe=lambda: next(values), on_sample=seen.append) as watch:
        pass
    assert len(seen) == 2
    assert watch.summary()["hostRssPeakMiB"] == 7
    assert watch.summary()["hostOomKills"] == 1


def test_failed_export_does_not_publish_a_partial_ply(tmp_path: Path) -> None:
    import trainer_export

    def fail(**kwargs: Any) -> bytes:
        raise RuntimeError("export failed")

    path = tmp_path / "out.ply"
    with pytest.raises(RuntimeError):
        trainer_export.write(
            fail, {"save_to": path, **{k: np.zeros(3) for k in trainer_export.FIELDS}}
        )
    assert not path.exists() and not path.with_suffix(".ply.part").exists()


def test_trainer_hook_disables_image_cache_and_only_streams_file_ply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import converge_trainer
    import trainer_export

    streamed: list[Any] = []
    direct: list[Any] = []

    def export(**kwargs: Any) -> bytes:
        direct.append(kwargs)
        return b"upstream"

    monkeypatch.setattr(trainer_export, "write", lambda real, kw: streamed.append((real, kw)))
    module: dict[str, Any] = {"export_splats": export}
    cfg = SimpleNamespace(cache_images=True, test_every=8)
    timing = converge_trainer.Timing(None)
    converge_trainer.instrument(module, None, timing, cfg)
    assert cfg.cache_images is False
    wrapped = module["export_splats"]
    assert wrapped(save_to="out.ply", format="ply") is None
    assert streamed == [(export, {"save_to": "out.ply", "format": "ply"})]
    assert wrapped(format="ply") == b"upstream"
    assert wrapped(save_to="out.splat", format="splat") == b"upstream"
    assert len(direct) == 2 and timing.plys == 3
