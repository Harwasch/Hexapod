"""The Modal image carries what the pipeline imports. Checked without `modal`.

`infra/modal/app.py`'s function body is `import remote, stages`, and `stages` imports
every stage module, so the container needs every third-party package any of them imports
at module level. The first version of that image installed two of them (boto3, numpy)
and none of `yaml`, `PIL` or `imageio_ffmpeg` -- an `ImportError` on the first GPU stage
anyone ran, found by replicating the image's venv, not by any test. This is that test:
the pipeline's own module-level imports, read with `ast`, against `IMAGE_PACKAGES`, read
out of `app.py` with `ast` too, so neither side needs `modal` installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import colmap4
import global_sfm
import sfm

PIPELINE = Path(__file__).resolve().parent.parent
APP = PIPELINE.parent.parent / "infra" / "modal" / "app.py"

#: Import name -> the distribution that provides it, where the two differ.
DISTRIBUTIONS = {"PIL": "pillow", "yaml": "pyyaml", "imageio_ffmpeg": "imageio-ffmpeg"}


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _image_packages() -> set[str]:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            if any(isinstance(t, ast.Name) and t.id == "IMAGE_PACKAGES" for t in targets):
                assert node.value is not None
                value = ast.literal_eval(node.value)
                return {spec.split(">")[0].split("=")[0].split("<")[0] for spec in value}
    raise AssertionError("app.py defines no IMAGE_PACKAGES")


def test_the_image_installs_every_package_the_pipeline_imports() -> None:
    local = {path.stem for path in PIPELINE.glob("*.py")}
    # The sibling project's modules, which the image copies beside the pipeline.
    local |= {path.stem for path in (PIPELINE.parent / "captures").glob("*.py")}
    third_party: set[str] = set()
    for path in PIPELINE.glob("*.py"):
        for name in _module_level_imports(path):
            if name in local or name in sys.stdlib_module_names or name == "__future__":
                continue
            third_party.add(DISTRIBUTIONS.get(name, name))

    missing = sorted(third_party - _image_packages())

    assert not missing, f"infra/modal/app.py IMAGE_PACKAGES lacks {missing}"


def test_the_image_puts_captures_beside_the_pipeline() -> None:
    """`captures_bridge` finds the packer at `../captures`; a copy anywhere else is an
    import error on the first stage, however complete the package list is."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    values = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"PIPELINE_DIR", "CAPTURES_DIR"}
    }

    assert Path(values["CAPTURES_DIR"]) == Path(values["PIPELINE_DIR"]).parent / "captures"


def _constants(*names: str) -> dict[str, object]:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    return {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }


def test_the_cpu_image_pins_the_vocabulary_tree_colmap_3_9_reads() -> None:
    """Loop detection needs the tree in the image `pose` runs in, at the path the
    pipeline looks for. The FLANN-format file, not 3.11's `vocab_tree_faiss_*`, which
    COLMAP 3.9.1 cannot load; and pinned by sha256, so a changed file fails the build."""
    values = _constants("VOCAB_TREE_URL", "VOCAB_TREE_SHA256", "VOCAB_TREE")
    url, digest, path = (
        str(values[k]) for k in ("VOCAB_TREE_URL", "VOCAB_TREE_SHA256", "VOCAB_TREE")
    )

    assert url.startswith("https://") and url.endswith("/vocab_tree_flickr100K_words32K.bin")
    assert Path(path).name == "vocab_tree_flickr100K_words32K.bin"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    source = APP.read_text(encoding="utf-8")
    assert f'"{sfm.VOCAB_TREE_ENV}": VOCAB_TREE' in source
    assert "sha256sum -c" in source


def test_the_cpu_image_pins_the_faiss_tree_colmap_4_2_reads_and_checks_it_loads() -> None:
    """`colmap: "4.2"` cannot read the FLANN tree above; it gets the faiss build of the
    same 32K-word tree, pinned by sha256, and the build loads it through pycolmap."""
    values = _constants("VOCAB_TREE_FAISS_URL", "VOCAB_TREE_FAISS_SHA256", "VOCAB_TREE_FAISS")
    url, digest, path = (
        str(values[k])
        for k in ("VOCAB_TREE_FAISS_URL", "VOCAB_TREE_FAISS_SHA256", "VOCAB_TREE_FAISS")
    )

    assert url.startswith("https://") and url.endswith("/vocab_tree_faiss_flickr100K_words32K.bin")
    assert Path(path).name == "vocab_tree_faiss_flickr100K_words32K.bin"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    source = APP.read_text(encoding="utf-8")
    assert f'"{colmap4.VOCAB_TREE_ENV}": VOCAB_TREE_FAISS' in source
    assert "sha256sum -c" in source
    assert "python colmap4.py self-check --vocab-tree {VOCAB_TREE_FAISS}" in source


def test_the_cpu_image_installs_the_pycolmap_the_global_mapper_was_measured_with() -> None:
    """`mapper: global` runs pycolmap in the image's own Python; a different version is a
    mapper nobody here has scored against the rendered orbit (global_sfm.py)."""
    pinned = str(_constants("PYCOLMAP")["PYCOLMAP"])

    assert pinned == f"pycolmap=={global_sfm.PYCOLMAP_VERSION}"
    assert ".pip_install(*IMAGE_PACKAGES, PYCOLMAP)" in APP.read_text(encoding="utf-8")


def test_gpu_functions_reserve_cpu_and_size_their_thread_pools_to_it() -> None:
    """A GPU function with no `cpu=` is reserved 0.125 of a core, and the trainer is fed
    by CPU: every GPU function (the fallback lists too) reserves cores and memory, and
    the image caps the thread pools that would otherwise size themselves to the host."""
    values = _constants("GPU_CPU_CORES", "GPU_MEMORY_MIB")
    cores, memory = float(str(values["GPU_CPU_CORES"])), int(str(values["GPU_MEMORY_MIB"]))
    assert cores >= 2.0 and memory >= 4096
    source = APP.read_text(encoding="utf-8")
    assert source.count("cpu=GPU_CPU_CORES") == 2 and source.count("memory=GPU_MEMORY_MIB") == 2
    assert "**THREAD_ENV" in source
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'"{variable}": "{int(2 * cores)}"' in source


def test_the_reservation_that_is_priced_is_the_one_that_is_deployed() -> None:
    """`costUsd` adds the GPU functions' CPU and memory reservation to the GPU's rate
    (`modal_adapter.reservation_rate`); a reservation changed in `app.py` and not there
    would be a cost that no longer matches what Modal bills."""
    from modal_adapter import GPU_RESERVATION

    values = _constants("GPU_CPU_CORES", "GPU_MEMORY_MIB")
    deployed = (float(str(values["GPU_CPU_CORES"])), int(str(values["GPU_MEMORY_MIB"])))
    assert deployed == GPU_RESERVATION


def test_colmap_is_capped_to_the_cpu_box_s_reservation() -> None:
    """COLMAP's -1 is every core the container can see -- the host's, dozens of them, each
    a SIFT thread holding a frame's scale space in 8 GiB. The CPU image says how many the
    `cpu4` reservation has, and the pose stage reads it (`sfm.default_threads`)."""
    from modal_adapter import CPU_TIERS

    threads = str(_constants("COLMAP_THREADS")["COLMAP_THREADS"])
    assert int(threads) == int(2 * CPU_TIERS["cpu4"][0])
    assert f'"{sfm.THREADS_ENV}": COLMAP_THREADS' in APP.read_text(encoding="utf-8")
    assert sfm.default_threads({sfm.THREADS_ENV: threads}) == 8
    assert sfm.default_threads({}) is None
    assert sfm.default_threads({sfm.THREADS_ENV: "lots"}) is None
    assert sfm.default_threads({sfm.THREADS_ENV: "0"}) is None
