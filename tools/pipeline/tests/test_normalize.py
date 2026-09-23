"""`normalize` / `ffmpeg_frames`, held to the three A0 findings it is built on.

The clip is generated here rather than committed: A9 took 104 MB out of git and a test
fixture that is a video file would start putting it back. Everything below runs from
frames this file drew, encoded with the same binary the stage uses.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image, ImageFilter

import video
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

#: Real `ffmpeg -i` output from an iPhone clip, kept verbatim: this is the only format
#: there is, since `imageio-ffmpeg` ships no ffprobe, and it is not an API.
IPHONE_STDERR = (
    "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'IMG_4417.MOV':\n"
    "  Metadata:\n"
    "    major_brand     : qt\n"
    "    minor_version   : 0\n"
    "    compatible_brands: qt\n"
    "    creation_time   : 2025-06-14T09:41:02.000000Z\n"
    "    com.apple.quicktime.location.accuracy.horizontal: 4.766775\n"
    "    com.apple.quicktime.location.ISO6709: +51.5074-000.1278+031.482/\n"
    "    com.apple.quicktime.make: Apple\n"
    "    com.apple.quicktime.model: iPhone 15 Pro\n"
    "    com.apple.quicktime.software: 18.5\n"
    "    com.apple.quicktime.creationdate: 2025-06-14T10:41:02+0100\n"
    "  Duration: 00:00:23.48, start: 0.000000, bitrate: 28871 kb/s\n"
    "  Stream #0:0[0x1](und): Video: hevc (Main 10) (hvc1 / 0x31637668), "
    "yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67), 3840x2160, 28633 kb/s, "
    "59.94 fps, 59.94 tbr, 600 tbn (default)\n"
    "      Metadata:\n"
    "        creation_time   : 2025-06-14T09:41:02.000000Z\n"
    "        handler_name    : Core Media Video\n"
    "        vendor_id       : [0][0][0][0]\n"
    "        encoder         : HEVC\n"
    "      Side data:\n"
    "        displaymatrix: rotation of -90.00 degrees\n"
)

#: The same clip written by something that is not the iOS camera: the older `(c)xyz`
#: atom only, which ffmpeg reports as `location`. Half the captures look like this.
CXYZ_STDERR = (
    "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':\n"
    "  Metadata:\n"
    "    location        : +37.7749-122.4194+010.000/\n"
    "    location-eng    : +37.7749-122.4194+010.000/\n"
    "  Duration: 00:00:04.00, start: 0.000000, bitrate: 900 kb/s\n"
    "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, "
    "1920x1080, 880 kb/s, 30 fps, 30 tbr, 15360 tbn (default)\n"
)


def _frame(index: int, *, blurred: bool, size: tuple[int, int] = (320, 240)) -> Image.Image:
    """A deterministic high-frequency pattern, optionally blurred.

    Seeded per frame so consecutive frames differ, which is what makes the clip a clip
    rather than one still repeated, and what stops the encoder collapsing it.
    """
    rng = np.random.default_rng(1000 + index)
    noise = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    # Blocks rather than per-pixel noise: per-pixel noise does not survive H.264, and a
    # frame whose sharpness the encoder destroyed would not be testing the selector.
    blocks = np.kron(noise[::8, ::8], np.ones((8, 8, 1), dtype=np.uint8))
    image = Image.fromarray(blocks[: size[1], : size[0]], mode="RGB")
    return image.filter(ImageFilter.GaussianBlur(radius=3.0)) if blurred else image


def make_clip(
    path: Path, *, frames: int = 20, fps: int = 10, metadata: list[str] | None = None
) -> Path:
    """Encode `frames` frames, every other one deliberately blurred, into `path`."""
    stills = path.parent / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        _frame(index, blurred=bool(index % 2)).save(stills / f"in_{index:04d}.png")
    argv = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(stills / "in_%04d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-pix_fmt",
        "yuv420p",
        *(metadata or []),
        str(path),
    ]
    subprocess.run(argv, capture_output=True, check=True)
    return path


def run_normalize(tmp_path: Path, params: dict[str, object], upload: Path) -> Workdir:
    workdir = Workdir.create(tmp_path / "run")
    target = workdir.input_path("upload")
    target.mkdir(parents=True, exist_ok=True)
    for entry in sorted(upload.iterdir()):
        if entry.is_file():
            (target / entry.name).write_bytes(entry.read_bytes())
    recipe = make_recipe(
        [{"id": "normalize", "impl": "ffmpeg_frames", "params": params}], inputs=["upload"]
    )
    execute(recipe, workdir, RunnerSet(cpu=LocalRunner()))
    return workdir


# --- A0 #6's trap: which ffmpeg ------------------------------------------------------


def test_the_stage_runs_the_wheels_ffmpeg_and_never_the_system_one(tmp_path: Path) -> None:
    """The one that stops this passing here and failing on ubuntu-latest.

    This machine has `/usr/bin/ffmpeg` and `ubuntu-latest` has none, so a stage that
    resolved the binary off PATH would be green locally and red in CI. The assertion is
    on the argv the stage actually ran, read back out of its own log.
    """
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(upload / "clip.mp4")

    workdir = run_normalize(tmp_path, {"fps": 10, "keep": 4}, upload)

    log = workdir.log_path("normalize").read_text()
    invocations = [line for line in log.splitlines() if line.startswith("$ ")]
    assert invocations, "the stage logged no subprocess at all"
    for line in invocations:
        binary = line[2:].split(" ", 1)[0]
        assert "imageio_ffmpeg" in binary, f"{binary} is not the binary that ships in the wheel"
        assert binary != "/usr/bin/ffmpeg"
        assert binary != "ffmpeg", "a bare name is resolved by PATH, which is the bug"
    assert Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()


def test_the_module_resolves_ffmpeg_in_exactly_one_place() -> None:
    """`video.ffmpeg_exe` is the only answer, and every argv builder goes through it."""
    exe = video.ffmpeg_exe()

    assert "imageio_ffmpeg" in Path(exe).parts
    assert video.probe_argv(Path("x.mp4"))[0] == exe
    assert video.extract_frames_argv(Path("x.mp4"), Path("f_%05d.jpg"), fps=2)[0] == exe


# --- A0 #6's finding: top-K, never a threshold ---------------------------------------


def test_sharpness_selection_keeps_the_sharp_half_of_a_clip(tmp_path: Path) -> None:
    """Twenty frames, every other one blurred, keep ten: the ten are the sharp ones.

    The gap assertion is the real one. Ten frames out of twenty is top-K by definition;
    what is being tested is that variance-of-Laplacian ranked the *blurred* frames last,
    which is A0's Spearman +0.979 restated as a fact about these frames.
    """
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(upload / "clip.mp4", frames=20, fps=10)

    workdir = run_normalize(tmp_path, {"fps": 10, "select": "sharpness", "keep": 10}, upload)

    frames = sorted((workdir.out_dir("normalize") / "frames").iterdir())
    assert len(frames) == 10
    assert [p.name for p in frames] == [f"frame_{i:04d}.jpg" for i in range(10)]
    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    kept = meta["sharpness"]["kept"]
    rejected = meta["sharpness"]["rejected"]
    assert meta["sharpness"]["metric"] == "variance-of-laplacian"
    assert "never an absolute cutoff" in meta["sharpness"]["selection"]
    # A clean separation, not a marginal one: the blurred half is far below the sharp
    # half, which is what makes the ranking a real signal rather than a coin toss.
    assert kept["min"] > 3.0 * rejected["max"], (kept, rejected)


def test_there_is_no_threshold_to_pass_even_by_accident(tmp_path: Path) -> None:
    """A0 measured a 101x within-clip range, so a cutoff cannot transfer between scenes.

    The stage refuses any `select` it does not implement rather than falling through to
    "keep everything", which is how a threshold would sneak back in as a default.
    """
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(upload / "clip.mp4", frames=4, fps=10)

    with pytest.raises(Exception, match="no blur threshold"):
        run_normalize(tmp_path, {"fps": 10, "select": "blur_threshold", "keep": 2}, upload)


def test_select_all_thins_evenly_rather_than_truncating() -> None:
    """Dropping the tail of an orbit loses a whole side of the object."""
    assert video.evenly_spaced(20, 5) == (0, 5, 10, 14, 19)
    assert video.evenly_spaced(4, 10) == (0, 1, 2, 3)
    assert video.evenly_spaced(10, 0) == ()


def test_top_k_returns_temporal_order_and_breaks_ties_on_the_earlier_frame() -> None:
    assert video.select_sharpest([5.0, 1.0, 4.0, 9.0], 2) == (0, 3)
    assert video.select_sharpest([1.0, 1.0, 1.0], 2) == (0, 1)
    assert video.select_sharpest([1.0, 2.0], 9) == (0, 1)


def test_windowed_selection_keeps_a_frame_from_a_blurred_stretch() -> None:
    """A sharp first half and a blurred second half, keep four of twelve.

    Global top-K keeps four frames of the first half and nothing of the second -- a
    whole side of an orbit gone. Windowed keeps the sharpest of each quarter, blurred or
    not, which is what COLMAP needs to register that side at all.
    """
    scores = [9.0, 8.0, 9.5, 8.5, 9.1, 8.2, 1.0, 1.2, 0.9, 1.1, 1.3, 0.8]

    assert video.select_sharpest(scores, 4) == (0, 2, 3, 4)
    assert video.select_sharpest_per_window(scores, 4) == (2, 4, 7, 10)


def test_windowed_selection_is_temporal_deterministic_and_bounded() -> None:
    assert video.select_sharpest_per_window([1.0, 1.0, 1.0, 1.0], 2) == (0, 2)
    assert video.select_sharpest_per_window([1.0, 2.0], 9) == (0, 1)
    assert video.select_sharpest_per_window([], 3) == ()
    assert video.select_sharpest_per_window([3.0, 1.0, 2.0], 0) == ()
    chosen = video.select_sharpest_per_window([float(i % 7) for i in range(241)], 100)
    assert len(chosen) == 100
    assert list(chosen) == sorted(set(chosen))


def test_the_windowed_mode_runs_in_the_stage_and_says_which_rule_chose(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(upload / "clip.mp4", frames=20, fps=10)

    workdir = run_normalize(
        tmp_path, {"fps": 10, "select": "sharpness-windowed", "keep": 10}, upload
    )

    assert len(list((workdir.out_dir("normalize") / "frames").iterdir())) == 10
    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frames"]["select"] == "sharpness-windowed"
    assert "equal stretches" in meta["sharpness"]["selection"]
    # Every other frame is blurred, so each window of two holds one sharp frame, and the
    # windowed rule keeps exactly the frames global top-K does.
    assert meta["sharpness"]["kept"]["min"] > 3.0 * meta["sharpness"]["rejected"]["max"]


def test_variance_of_laplacian_ranks_a_blurred_frame_below_its_sharp_original(
    tmp_path: Path,
) -> None:
    sharp, blurred = tmp_path / "s.png", tmp_path / "b.png"
    _frame(3, blurred=False).save(sharp)
    _frame(3, blurred=True).save(blurred)

    assert video.sharpness(sharp) > 10.0 * video.sharpness(blurred)


# --- A0's sizing note: both iPhone location keys -------------------------------------


def test_the_mdta_iso6709_key_is_read() -> None:
    meta = video.parse_probe(IPHONE_STDERR)

    assert meta.location is not None
    assert meta.location.source == "com.apple.quicktime.location.iso6709"
    assert (round(meta.location.lat, 4), round(meta.location.lon, 4)) == (51.5074, -0.1278)
    assert meta.location.altitude_m == pytest.approx(31.482)


def test_the_older_cxyz_atom_is_read_too() -> None:
    """ffmpeg reports the `(c)xyz` atom as `location`; reading only the mdta key
    would silently lose the location of every clip not written by the iOS camera."""
    meta = video.parse_probe(CXYZ_STDERR)

    assert meta.location is not None
    assert meta.location.source == "location"
    assert (meta.location.lat, meta.location.lon) == (37.7749, -122.4194)


def test_the_rest_of_the_iphone_header_is_scraped_too() -> None:
    meta = video.parse_probe(IPHONE_STDERR)

    assert meta.make == "Apple"
    assert meta.model == "iPhone 15 Pro"
    assert meta.software == "18.5"
    assert meta.created_at == "2025-06-14T10:41:02+0100"
    assert (meta.width, meta.height) == (3840, 2160)
    assert meta.codec == "hevc"
    assert meta.fps == pytest.approx(59.94)
    assert meta.duration_s == pytest.approx(23.48)
    # A portrait clip is landscape pixels plus this, and losing it is losing the
    # orientation every frame is then extracted in.
    assert meta.rotation_deg == pytest.approx(-90.0)


def test_a_header_with_nothing_in_it_yields_nulls_rather_than_raising() -> None:
    """Scraped stderr is not an API; the parser is total on purpose."""
    meta = video.parse_probe("ffmpeg version 7.0.2\nnothing here resembles a container\n")

    assert meta.location is None
    assert meta.duration_s is None and meta.width is None and meta.make is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("+37.7749-122.4194+010.000/", (37.7749, -122.4194, 10.0)),
        ("+51.5074-000.1278+031.482/", (51.5074, -0.1278, 31.482)),
        ("-33.8688+151.2093/", (-33.8688, 151.2093, None)),
    ],
)
def test_iso6709_signs_are_the_separators(
    text: str, expected: tuple[float, float, float | None]
) -> None:
    parsed = video.parse_iso6709(text)
    assert parsed is not None
    assert (round(parsed[0], 4), round(parsed[1], 4), parsed[2]) == expected


@pytest.mark.parametrize("text", ["", "51.5074,-0.1278", "unknown", "+51.5074/"])
def test_a_coordinate_nobody_can_vouch_for_is_none_rather_than_a_guess(text: str) -> None:
    assert video.parse_iso6709(text) is None


# --- the whole stage, on a real clip -------------------------------------------------


def test_an_iphone_portrait_hevc_mov_comes_out_upright_with_its_location(
    tmp_path: Path,
) -> None:
    """The shape an iPhone held upright writes, built rather than downloaded: HEVC in a
    QuickTime container, the pixels stored *landscape* with a display matrix that turns
    them clockwise on playback, and the `mdta` location key. The frames must come out
    portrait and the right way up -- ffmpeg honours the matrix by default, and this is
    what notices if that ever stops being the default or the argv turns it off.

    The same check was run on 100 real photographs encoded this way (1080x1920, from
    nerfstudio's `dozer` capture): 100/100 upright.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    upright = Image.new("RGB", (240, 320), (0, 0, 0))
    upright.paste((255, 255, 255), (0, 0, 240, 80))  # the top quarter is white
    stills = tmp_path / "stills"
    stills.mkdir()
    for index in range(4):
        # Stored as the sensor reads out in portrait: a quarter turn counter-clockwise.
        upright.transpose(Image.Transpose.ROTATE_90).save(stills / f"s_{index:04d}.png")
    raw = tmp_path / "raw.mov"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-y", "-framerate", "4", "-i",
         str(stills / "s_%04d.png"), "-c:v", "libx265", "-x265-params", "log-level=error",
         "-pix_fmt", "yuv420p", "-tag:v", "hvc1", str(raw)],
        capture_output=True, check=True,
    )  # fmt: skip
    upload = tmp_path / "upload"
    upload.mkdir()
    subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-y", "-display_rotation:v:0", "-90",
         "-i", str(raw), "-c", "copy", "-movflags", "use_metadata_tags",
         "-metadata", "com.apple.quicktime.location.ISO6709=+37.7694-122.4862+012.000/",
         "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
         str(upload / "IMG_0001.MOV")],
        capture_output=True, check=True,
    )  # fmt: skip

    workdir = run_normalize(tmp_path, {"fps": 4, "keep": 4, "select": "all"}, upload)

    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["video"]["codec"] == "hevc"
    assert meta["video"]["rotationDeg"] == -90.0
    assert (meta["frames"]["width"], meta["frames"]["height"]) == (240, 320)
    assert meta["location"]["source"] == "com.apple.quicktime.location.iso6709"
    assert (meta["location"]["lat"], meta["location"]["lon"]) == (37.7694, -122.4862)
    assert meta["device"] == "iPhone 15 Pro"
    for frame in sorted((workdir.out_dir("normalize") / "frames").iterdir()):
        grey = np.asarray(Image.open(frame).convert("L"), dtype=np.float64)
        assert grey.shape == (320, 240)
        # White on top, black at the bottom: upright, not turned or flipped.
        assert grey[:60].mean() > 200 and grey[-60:].mean() < 50


def test_both_location_keys_survive_a_round_trip_through_a_real_container(
    tmp_path: Path,
) -> None:
    """Written by ffmpeg, read back by the stage. Not a string fixture: a file."""
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(
        upload / "clip.mp4",
        frames=4,
        fps=10,
        metadata=["-metadata", "location=+37.7749-122.4194+010.000/"],
    )

    workdir = run_normalize(tmp_path, {"fps": 10, "keep": 2}, upload)

    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["location"]["source"] == "location"
    assert (meta["location"]["lat"], meta["location"]["lon"]) == (37.7749, -122.4194)


def test_a_clip_becomes_frames_and_a_description_of_where_they_came_from(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(
        upload / "clip.mp4",
        frames=20,
        fps=10,
        metadata=[
            "-movflags",
            "use_metadata_tags",
            "-metadata",
            "com.apple.quicktime.make=Apple",
            "-metadata",
            "com.apple.quicktime.model=iPhone 15 Pro",
            "-metadata",
            "com.apple.quicktime.location.ISO6709=+51.5074-000.1278+031.482/",
        ],
    )

    workdir = run_normalize(
        tmp_path, {"fps": 5, "select": "sharpness", "keep": 6, "captured_at": "2025-06-14"}, upload
    )

    out = workdir.out_dir("normalize")
    meta = json.loads((out / "source_meta.json").read_text())
    assert meta["format"] == "video"
    assert meta["filename"] == "clip.mp4"
    assert meta["checksum"].startswith("sha256:")
    assert meta["frames"]["kept"] == 6
    assert meta["frames"]["fps"] == 5
    assert meta["frames"]["width"] == 320 and meta["frames"]["height"] == 240
    assert meta["video"]["codec"] == "h264"
    assert meta["device"] == "iPhone 15 Pro"
    assert meta["sensor"] == "Apple"
    # The capture row wins over the container, which is the same rule ingest_splat uses.
    assert meta["capturedAt"] == "2025-06-14"
    assert meta["location"]["source"] == "com.apple.quicktime.location.iso6709"
    assert meta["tools"]["ffmpegFrom"] == "imageio-ffmpeg"
    step = json.loads(workdir.step_path("normalize").read_text())
    assert step["metrics"]["frames"] == 6
    assert step["metrics"]["select"] == "sharpness"


def test_a_folder_of_stills_is_a_capture_too(tmp_path: Path) -> None:
    """No video, no ffmpeg extraction -- the same artifact contract out the other end."""
    upload = tmp_path / "upload"
    upload.mkdir()
    for index in range(6):
        _frame(index, blurred=bool(index % 2)).save(upload / f"IMG_{index:04d}.jpg")

    workdir = run_normalize(tmp_path, {"select": "sharpness", "keep": 3}, upload)

    out = workdir.out_dir("normalize")
    meta = json.loads((out / "source_meta.json").read_text())
    assert meta["format"] == "images"
    assert meta["video"] is None and meta["frames"]["fps"] is None
    assert sorted(p.name for p in (out / "frames").iterdir()) == [
        "frame_0000.jpg",
        "frame_0001.jpg",
        "frame_0002.jpg",
    ]


def test_an_upload_with_nothing_in_it_says_what_it_was_looking_for(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "notes.txt").write_text("no frames here")

    with pytest.raises(Exception, match="holds no video"):
        run_normalize(tmp_path, {"keep": 2}, upload)


def test_a_video_beside_a_poster_frame_is_still_a_video(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    make_clip(upload / "clip.mp4", frames=4, fps=10)
    _frame(0, blurred=False).save(upload / "poster.jpg")

    assert video.pick_source(upload).kind == "video"
