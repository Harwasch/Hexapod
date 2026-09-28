"""`normalize` / `ffmpeg_frames`, held to the three A0 findings it is built on.

The clip is generated here rather than committed: A9 took 104 MB out of git and a test
fixture that is a video file would start putting it back. Everything below runs from
frames this file drew, encoded with the same binary the stage uses.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image, ImageFilter

import keyframes
import resolution
import video
from conftest import make_recipe
from executor import execute
from runners import LocalRunner, RunnerSet
from workdir import Workdir

PIPELINE = Path(__file__).resolve().parent.parent

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


# --- frame size --------------------------------------------------------------------


def test_a_full_size_photo_is_shrunk_and_keeps_its_gps(tmp_path: Path) -> None:
    """An iPhone photo is 5712x4284; training on that multiplied the GPU time for no
    detail a splat holds. The GPS has to survive, because `georeference` reads it."""
    import gps_frames
    from PIL import Image

    import exif

    source = tmp_path / "in" / "IMG_0001.jpeg"
    source.parent.mkdir()
    Image.new("RGB", (3000, 2000), (90, 140, 60)).save(source, quality=90)
    gps_frames.write_gps(source, 44.9778, -93.265, 256.0)

    (out,) = video.copy_frames([source], tmp_path / "out", max_side=1600)
    with Image.open(out) as image:
        assert image.size == (1600, 1067)
    fix = exif.read_fix(out)
    assert fix is not None
    assert (round(fix.lat, 4), round(fix.lon, 4)) == (44.9778, -93.265)


def test_a_frame_already_small_enough_is_copied_byte_for_byte(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "small.jpg"
    Image.new("RGB", (1280, 720), (10, 20, 30)).save(source, quality=90)
    (out,) = video.copy_frames([source], tmp_path / "out", max_side=1600)
    assert out.read_bytes() == source.read_bytes()


def test_video_frames_are_bounded_on_the_long_side_even_in_portrait() -> None:
    """The filter once bounded the width only, so a portrait clip -- width its short
    side -- kept nearly its full height."""
    argv = video.extract_frames_argv(Path("x.mov"), Path("f_%05d.jpg"), fps=4, max_side=1600)
    chain = argv[argv.index("-vf") + 1]
    assert "if(gte(iw,ih),min(1600,iw),-2)" in chain
    assert "if(gte(iw,ih),-2,min(1600,ih))" in chain


def test_a_video_is_capped_by_keep_video_and_photos_by_keep(tmp_path: Path) -> None:
    """Sequential matching lets a video keep more frames than an exhaustively matched set."""
    clip = tmp_path / "clip"
    clip.mkdir()
    make_clip(clip / "clip.mp4")
    workdir = run_normalize(tmp_path / "v", {"fps": 10, "keep": 2, "keep_video": 5}, clip)
    assert len(list((workdir.out_dir("normalize") / "frames").iterdir())) == 5

    stills = tmp_path / "stills"
    stills.mkdir()
    for index in range(6):
        _frame(index, blurred=False).save(stills / f"IMG_{index:04d}.jpg")
    workdir = run_normalize(tmp_path / "p", {"keep": 2, "keep_video": 5}, stills)
    assert len(list((workdir.out_dir("normalize") / "frames").iterdir())) == 2


# --- select: viewpoint -- keyframes by camera motion ---------------------------------


def _world(width: int, height: int, *, seed: int = 7) -> np.ndarray:
    """A grey world wider than any frame, textured at several scales so that phase
    correlation has something to lock onto in every tile and H.264 keeps it."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(height // 8 + 1, width // 8 + 1), dtype=np.uint8)
    blocks = np.kron(coarse, np.ones((8, 8), dtype=np.uint8))[:height, :width]
    image = Image.fromarray(blocks, mode="L").filter(ImageFilter.GaussianBlur(radius=1.0))
    return np.asarray(image, dtype=np.float64)


def _encode(stills: Path, path: Path, fps: int) -> Path:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
         "-framerate", str(fps), "-i", str(stills / "in_%04d.png"),
         "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )  # fmt: skip
    return path


def make_pan_clip(
    path: Path,
    *,
    offsets: list[float],
    size: tuple[int, int] = (320, 240),
    lower_offsets: list[float] | None = None,
    blur_every_other: bool = False,
    fps: int = 10,
) -> Path:
    """A camera panning over a flat world: frame i is the window at `offsets[i]` px.

    `lower_offsets`, when given, moves the lower half of the picture by its own amount --
    a nearer surface sliding past a farther one, which is parallax: no single homography
    maps one frame onto the next.
    """
    width, height = size
    reach = int(max(offsets + (lower_offsets or [0.0]))) + width + 2
    world = _world(reach, height)
    far = _world(reach, height, seed=11)
    stills = path.parent / f"stills_{path.stem}"
    stills.mkdir(parents=True, exist_ok=True)
    for index, offset in enumerate(offsets):
        x = round(offset)
        frame = world[:, x : x + width].copy()
        if lower_offsets is not None:
            lower = round(lower_offsets[index])
            frame[height // 2 :] = far[height // 2 :, lower : lower + width]
        image = Image.fromarray(frame.astype(np.uint8), mode="L").convert("RGB")
        if blur_every_other and index % 2:
            image = image.filter(ImageFilter.GaussianBlur(radius=3.0))
        image.save(stills / f"in_{index:04d}.png")
    return _encode(stills, path, fps)


def _viewpoint(tmp_path: Path, clip: Path, **params: object) -> dict[str, Any]:
    upload = tmp_path / f"upload_{clip.stem}"
    upload.mkdir(parents=True)
    (upload / clip.name).write_bytes(clip.read_bytes())
    settings: dict[str, object] = {"select": "viewpoint", "fps": 10, "keep_video": 300}
    settings.update(params)
    workdir = run_normalize(tmp_path / f"run_{clip.stem}", settings, upload)
    out = workdir.out_dir("normalize")
    meta: dict[str, Any] = json.loads((out / "source_meta.json").read_text())
    frames = sorted((workdir.out_dir("normalize") / "frames").iterdir())
    assert meta["frames"]["kept"] == len(frames)
    return meta


def test_a_fast_pan_keeps_more_frames_than_a_slow_one_over_the_same_time(
    tmp_path: Path,
) -> None:
    """The point of the mode: frames follow the view covered, not the seconds spent.

    Forty frames at 10 fps either way. The slow pan crosses 12% of a frame, about one
    window of the 10% overlap budget; the fast pan crosses 1.5 frames, ~15 windows. By
    time the two would keep the same number."""
    slow = make_pan_clip(tmp_path / "slow.mp4", offsets=[1.0 * i for i in range(40)])
    fast = make_pan_clip(tmp_path / "fast.mp4", offsets=[12.0 * i for i in range(40)])

    slow_meta = _viewpoint(tmp_path, slow)
    fast_meta = _viewpoint(tmp_path, fast)

    assert slow_meta["frames"]["candidates"] == fast_meta["frames"]["candidates"] == 40
    assert slow_meta["frames"]["kept"] <= 3, slow_meta["viewpoint"]
    assert 11 <= fast_meta["frames"]["kept"] <= 20, fast_meta["viewpoint"]
    view = fast_meta["viewpoint"]
    assert view["closedBy"].get("overlap", 0) >= 10
    assert view["unmeasuredPairs"] == 0
    assert not view["ceilingBound"]
    # What the budget promises between kept neighbours: ~90%, never much under 80%.
    assert view["neighbourOverlap"]["min"] >= 0.75
    assert 0.8 <= view["neighbourOverlap"]["median"] <= 0.97


def test_a_camera_that_does_not_move_keeps_almost_nothing(tmp_path: Path) -> None:
    """Near-duplicates are what time-based sampling filled a slow close-up with."""
    still = make_pan_clip(tmp_path / "still.mp4", offsets=[40.0] * 40)

    meta = _viewpoint(tmp_path, still)

    assert meta["frames"]["candidates"] == 40
    assert meta["frames"]["kept"] <= 2, meta["viewpoint"]


def test_parallax_closes_windows_that_overlap_alone_would_not(tmp_path: Path) -> None:
    """Near things sliding past far ones is a change of viewpoint even when the frame's
    content barely changes -- the case a pure overlap rule would under-sample, since
    circling a subject keeps it in view. The same slow pan twice; in the second the lower
    half moves at twice the speed, as a nearer surface would."""
    offsets = [1.5 * i for i in range(40)]
    flat = make_pan_clip(tmp_path / "flat.mp4", offsets=offsets)
    deep = make_pan_clip(
        tmp_path / "deep.mp4", offsets=offsets, lower_offsets=[3.0 * i for i in range(40)]
    )

    flat_meta = _viewpoint(tmp_path, flat)
    deep_meta = _viewpoint(tmp_path, deep)

    assert deep_meta["frames"]["kept"] >= flat_meta["frames"]["kept"] + 4
    assert deep_meta["viewpoint"]["closedBy"].get("parallax", 0) >= 4


def test_within_a_window_the_sharpest_candidate_is_the_one_kept(tmp_path: Path) -> None:
    """A0's rank still decides, inside each window of motion: every other frame of this
    pan is blurred, and not one of them is kept."""
    clip = make_pan_clip(
        tmp_path / "blurry.mp4", offsets=[6.0 * i for i in range(40)], blur_every_other=True
    )

    meta = _viewpoint(tmp_path, clip)

    assert 5 <= meta["frames"]["kept"] <= 20, meta["viewpoint"]
    assert meta["viewpoint"]["candidatesPerKeyframe"]["min"] >= 2, meta["viewpoint"]
    # The blurred frames score far below the sharp ones (A0 #6's 101x range), so the
    # least sharp kept frame towers over the least sharp candidate, which is blurred.
    assert meta["sharpness"]["kept"]["min"] > 3.0 * meta["sharpness"]["rejected"]["min"]
    assert "window of camera motion" in meta["sharpness"]["selection"]


def test_the_ceiling_spreads_keyframes_evenly_by_motion_not_by_time(tmp_path: Path) -> None:
    """Twenty frames of a crawl, then twenty of a fast pan, and a ceiling of six. Thinning
    by time would keep three of each half; by motion the crawl covered almost nothing, so
    the six are spread over the pan."""
    offsets = [0.5 * i for i in range(20)] + [10.0 + 14.0 * i for i in range(20)]
    clip = make_pan_clip(tmp_path / "mixed.mp4", offsets=offsets)

    meta = _viewpoint(tmp_path, clip, keep_video=6)

    view = meta["viewpoint"]
    assert meta["frames"]["kept"] == 6, view
    assert view["ceilingBound"] is True
    assert view["windowsBeforeCeiling"] > 6
    assert view["scale"] > 1.0
    # The crawl is one window at most; the rest of the six share the pan evenly.
    assert view["candidatesPerKeyframe"]["max"] >= 15


def test_the_ceiling_bisection_keeps_the_count_at_or_under_it() -> None:
    """Unit-level: pair motions made by hand, slow then fast, against several ceilings."""
    size = (480, 270)

    def shift(dx: float) -> keyframes.PairMotion:
        homography = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return keyframes.PairMotion(homography, np.zeros((7, 14, 2)), 60, True)

    pairs = [None, *([shift(2.0)] * 100), *([shift(20.0)] * 100)]
    free = keyframes.segment(pairs, frame_size=size, overlap=0.9, parallax=0.005)
    # 200 px of crawl and 2,000 of pan; a window closes at the first candidate 48 px (10%
    # of 480) from its start: 4 windows of 24 crawling candidates, ~34 of 3 panning ones.
    assert not free.bound
    assert 35 <= len(free.windows) <= 42
    for ceiling in (12, 20, 30):
        capped = keyframes.segment(
            pairs, frame_size=size, overlap=0.9, parallax=0.005, ceiling=ceiling
        )
        assert capped.bound
        # At or a little under it: windows are whole candidates, so the count moves in
        # steps as the budgets widen, and a pan of 20 px a candidate steps coarsely.
        assert 0.8 * ceiling <= len(capped.windows) <= ceiling
        whole = [w for w in capped.windows if w.closed_by != "end"]
        slow = [w.end - w.start + 1 for w in whole if w.end <= 100]
        fast = [w.end - w.start + 1 for w in whole if w.start > 100]
        # A crawling window holds ~10x the candidates of a panning one: equal motion.
        assert slow and fast
        assert min(slow) > 3 * max(fast)


def test_select_per_window_takes_each_windows_sharpest_and_breaks_ties_early() -> None:
    windows = [
        keyframes.Window(0, 2, 1.0, "overlap"),
        keyframes.Window(3, 3, 1.0, "parallax"),
        keyframes.Window(4, 6, 0.3, "end"),
    ]
    assert keyframes.select_per_window([1.0, 5.0, 2.0, 0.1, 3.0, 3.0, 1.0], windows) == (1, 3, 4)


def test_an_unmeasurable_pair_counts_as_a_whole_window() -> None:
    """When unsure, keep a frame: a gap pose cannot bridge costs more than a spare frame."""
    still = keyframes.PairMotion(np.eye(3), np.zeros((2, 2, 2)), 50, True)
    lost = keyframes.PairMotion(np.eye(3), np.full((2, 2, 2), np.nan), 3, False)
    windows = keyframes.segment_within(
        [None, still, still, lost, still, still], frame_size=(480, 270), overlap=0.9,
        parallax=0.005,
    )  # fmt: skip
    assert [(w.start, w.end) for w in windows] == [(0, 2), (3, 5)]
    assert windows[0].closed_by == "unmeasured"


def test_a_frame_that_matches_nothing_is_bridged_rather_than_breaking_the_window() -> None:
    """A burst of blur or a hand across the lens: the next frame is matched across it."""
    world = _world(700, 300).astype(np.float32)
    lens_cap = np.full((270, 480), 20.0, dtype=np.float32)
    tracker = keyframes.MotionTracker()
    pairs = [
        tracker.push(world[:270, 0:480]),
        tracker.push(world[:270, 4:484]),
        tracker.push(lens_cap),
        tracker.push(world[:270, 12:492]),
    ]
    assert pairs[0] is None
    assert pairs[1] is not None and pairs[1].measured
    assert pairs[2] is not None and pairs[2].held and not pairs[2].measured
    bridged = pairs[3]
    assert bridged is not None and bridged.measured
    # Measured from the last frame before the gap: 8 px, not 4.
    assert bridged.homography[0, 2] == pytest.approx(-8.0, abs=0.15)
    segmentation = keyframes.segment(pairs, frame_size=(480, 270), overlap=0.9, parallax=0.005)
    assert segmentation.unmeasured_pairs == 0
    assert len(segmentation.windows) == 1


def test_overlap_is_symmetric_in_zoom_and_exact_for_a_pan() -> None:
    pan = np.array([[1.0, 0.0, 48.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert keyframes.overlap_of(pan, 480, 270) == pytest.approx(0.9)
    assert keyframes.overlap_of(np.diag([1.25, 1.25, 1.0]), 480, 270) == pytest.approx(0.64)
    assert keyframes.overlap_of(np.diag([0.8, 0.8, 1.0]), 480, 270) == pytest.approx(0.64)
    assert keyframes.overlap_of(np.eye(3), 480, 270) == pytest.approx(1.0)


def test_the_tracker_recovers_a_known_shift_to_a_tenth_of_a_pixel() -> None:
    world = _world(700, 400).astype(np.float32)
    before = world[40:310, 60:540]
    after = world[33:303, 48:528]  # the content moves 12 px right and 7 px down
    motion = keyframes.pair_motion(before, after)
    assert motion.measured and motion.tiles >= 40
    assert motion.homography[0, 2] == pytest.approx(12.0, abs=0.1)
    assert motion.homography[1, 2] == pytest.approx(7.0, abs=0.1)
    assert np.nanmax(np.abs(motion.residuals)) < 0.3


def test_candidates_are_bounded_by_the_clips_rate_and_by_max_candidates(
    tmp_path: Path,
) -> None:
    """`fps` is the candidate rate under `viewpoint`, never above the clip's own (ffmpeg
    would duplicate frames to reach it) and never more than `max_candidates` in all."""
    clip = make_pan_clip(tmp_path / "rate.mp4", offsets=[3.0 * i for i in range(40)])

    meta = _viewpoint(tmp_path / "a", clip, fps=30)
    assert meta["frames"]["fps"] == 10.0
    assert meta["frames"]["fpsRequested"] == 30.0
    assert meta["frames"]["candidates"] == 40

    meta = _viewpoint(tmp_path / "b", clip, fps=30, max_candidates=20)
    assert meta["frames"]["fps"] == pytest.approx(5.0)
    assert meta["frames"]["candidates"] == 20


def test_a_photo_set_keeps_todays_rule_under_viewpoint(tmp_path: Path) -> None:
    """No order to measure motion along, and exhaustive (quadratic) matching: photos stay
    top-K by sharpness, windowed, capped by `keep`."""
    upload = tmp_path / "upload"
    upload.mkdir()
    for index in range(8):
        _frame(index, blurred=bool(index % 2)).save(upload / f"IMG_{index:04d}.jpg")

    workdir = run_normalize(tmp_path, {"select": "viewpoint", "keep": 4}, upload)

    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frames"]["select"] == "sharpness-windowed"
    assert meta["frames"]["selectRequested"] == "viewpoint"
    assert meta["frames"]["kept"] == 4
    assert meta["viewpoint"] is None


def test_the_recipe_selects_by_viewpoint_and_sizes_automatically() -> None:
    from recipe import load_recipe

    recipe = load_recipe(PIPELINE / "recipes" / "photo-reconstruct.yaml")
    params = next(stage for stage in recipe.stages if stage.id == "normalize").params
    assert params["select"] == "viewpoint"
    assert params["max_side"] == "auto"
    # The ceiling is well above the spool's 173 frames, so it is not the usual limit.
    assert params["keep_video"] >= 250
    assert params["fps"] > 8


# --- max_side: auto -- frame size from measured detail (#6) --------------------------


def _detailed(size: tuple[int, int], *, seed: int = 3) -> Image.Image:
    """Fine texture over most of the picture and a flat sky over the rest: real detail
    well above 1600 px, and a quiet region to read the noise floor from."""
    width, height = size
    rng = np.random.default_rng(seed)
    fine = rng.integers(0, 256, size=(height // 2 + 1, width // 2 + 1), dtype=np.uint8)
    texture = np.kron(fine, np.ones((2, 2), dtype=np.uint8))[:height, :width].astype(np.float64)
    texture = 0.5 * texture + 64.0
    gradient = np.linspace(150.0, 190.0, width)[None, :].repeat(height, axis=0)
    picture = np.where(np.arange(height)[:, None] < height * 0.3, gradient, texture)
    image = Image.fromarray(picture.astype(np.uint8), mode="L")
    return image.filter(ImageFilter.GaussianBlur(radius=0.6)).convert("RGB")


def _upscaled(size: tuple[int, int]) -> Image.Image:
    """The same kind of picture holding only 1000 px of it: shrunk, then enlarged back."""
    width, height = size
    small = _detailed(size).resize((1000, round(1000 * height / width)), Image.Resampling.LANCZOS)
    return small.resize(size, Image.Resampling.BICUBIC)


def _grey(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def test_detail_above_1600_is_measured_where_it_exists_and_only_there() -> None:
    size = (2400, 1350)
    detailed = resolution.decide([_grey(_detailed(size, seed=s)) for s in (1, 2)], source_side=2400)
    upscaled = resolution.decide([_grey(_upscaled(size))], source_side=2400)
    blurred = resolution.decide(
        [_grey(_detailed(size).filter(ImageFilter.GaussianBlur(radius=3.0)))], source_side=2400
    )

    assert detailed.max_side == 2400, detailed.reason
    assert detailed.detail_share is not None and detailed.detail_share > 0.5
    assert upscaled.max_side == 1600, upscaled.reason
    assert blurred.max_side == 1600, blurred.reason
    assert blurred.detail_share is not None and blurred.detail_share < 0.05


def test_noise_is_not_mistaken_for_detail() -> None:
    """A grainy clip carries energy above 1600 px everywhere, the quiet sky included; the
    floor read from the quiet tenth of the picture is what keeps that at 1600."""
    size = (2400, 1350)
    grainy = _grey(_upscaled(size)) + np.random.default_rng(0).normal(0, 5, (1350, 2400))
    decision = resolution.decide([np.clip(grainy, 0, 255).astype(np.float32)], source_side=2400)
    assert decision.max_side == 1600, decision.reason


def _textured(size: tuple[int, int], *, seed: int) -> Image.Image:
    """`_detailed` with no sky: texture edge to edge, as a tree filling the frame is."""
    width, height = size
    rng = np.random.default_rng(seed)
    fine = rng.integers(0, 256, size=(height // 2 + 1, width // 2 + 1), dtype=np.uint8)
    texture = np.kron(fine, np.ones((2, 2), dtype=np.uint8))[:height, :width].astype(np.float64)
    image = Image.fromarray((0.5 * texture + 64.0).astype(np.uint8), mode="L")
    return image.filter(ImageFilter.GaussianBlur(radius=0.6)).convert("RGB")


def test_the_noise_floor_is_the_captures_not_each_frames() -> None:
    """A frame textured edge to edge has no quiet region, so on its own it reads its
    texture as noise and earns nothing (the Minnetonka orbit: three frames of foliage and
    lawn at floors of 4-5 grey levels kept 1600 px). The camera's noise shows in the
    sample's quietest frame; read against it, the texture is detail."""
    size = (2400, 1350)
    textured = [_grey(_textured(size, seed=s)) for s in (1, 2, 3)]
    alone = resolution.decide(textured, source_side=2400)
    assert alone.max_side == 1600, alone.reason  # no frame shows the camera's noise
    with_sky = resolution.decide([*textured, _grey(_detailed(size, seed=4))], source_side=2400)
    assert with_sky.max_side == 2400, with_sky.reason
    assert "capture's noise floor" in with_sky.reason
    assert resolution.capture_noise_floor([*textured, _grey(_detailed(size, seed=4))]) < 1.0


def test_a_blown_out_sky_is_not_a_quiet_camera() -> None:
    """Clipped highlights carry no noise because they carry no signal: a grainy clip with
    a white sky must not read the sky's zero as the camera's floor."""
    size = (2400, 1350)
    rng = np.random.default_rng(0)
    grainy = []
    for _ in range(2):
        frame = _grey(_upscaled(size)) + rng.normal(0, 5, (1350, 2400))
        frame[: int(1350 * 0.3)] = 255.0  # the sky, blown out
        grainy.append(np.clip(frame, 0, 255).astype(np.float32))
    decision = resolution.decide(grainy, source_side=2400)
    assert decision.max_side == 1600, decision.reason
    assert resolution.capture_noise_floor(grainy) > 2.0


def test_a_source_too_small_to_gain_is_not_measured() -> None:
    decision = resolution.decide([], source_side=1920)
    assert decision.max_side == 1600
    assert decision.measured_side is None
    assert "1920" in decision.reason


def test_auto_size_keeps_detailed_photos_large_and_soft_ones_at_1600(tmp_path: Path) -> None:
    sharp, soft = tmp_path / "sharp", tmp_path / "soft"
    sharp.mkdir()
    soft.mkdir()
    for index in range(2):
        _detailed((3000, 2000), seed=index).save(sharp / f"IMG_{index:04d}.jpg", quality=95)
        _upscaled((3000, 2000)).save(soft / f"IMG_{index:04d}.jpg", quality=95)

    params = {"select": "all", "keep": 2, "max_side": "auto"}
    kept = run_normalize(tmp_path / "a", params, sharp)
    shrunk = run_normalize(tmp_path / "b", params, soft)

    for workdir, side in ((kept, 2400), (shrunk, 1600)):
        meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
        assert meta["frameSize"]["rule"] == "auto"
        assert meta["frameSize"]["maxSide"] == side, meta["frameSize"]["reason"]
        assert meta["frameSize"]["measuredSide"] == 2400
        assert meta["frameSize"]["frames"], "the measurement is recorded, not just its answer"
        for frame in (workdir.out_dir("normalize") / "frames").iterdir():
            with Image.open(frame) as image:
                assert max(image.size) == side


def test_auto_size_on_a_video_samples_it_and_extracts_at_the_answer(tmp_path: Path) -> None:
    """The video path: seeks at the measured size, then extraction at the answer."""
    stills = tmp_path / "stills"
    stills.mkdir()
    for index in range(3):
        _detailed((2560, 1440), seed=index).save(stills / f"in_{index:04d}.png", compress_level=1)
    upload = tmp_path / "upload"
    upload.mkdir()
    _encode(stills, upload / "detailed.mp4", 2)

    workdir = run_normalize(
        tmp_path / "run", {"select": "all", "fps": 2, "keep": 2, "max_side": "auto"}, upload
    )

    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frameSize"]["sourceSide"] == 2560
    assert meta["frameSize"]["maxSide"] == 2400, meta["frameSize"]["reason"]
    assert (meta["frames"]["width"], meta["frames"]["height"]) == (2400, 1350)
    log = workdir.log_path("normalize").read_text()
    assert log.count(" -ss ") >= 4, "the size sample is seeks, not a decode of the clip"


def test_an_explicit_max_side_overrides_auto_and_nonsense_is_refused(tmp_path: Path) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    _detailed((3000, 2000)).save(upload / "IMG_0000.png")

    workdir = run_normalize(tmp_path / "a", {"select": "all", "max_side": 1200}, upload)
    meta = json.loads((workdir.out_dir("normalize") / "source_meta.json").read_text())
    assert meta["frameSize"] == {"rule": "fixed", "maxSide": 1200}
    assert meta["frames"]["width"] == 1200

    with pytest.raises(Exception, match="max_side"):
        run_normalize(tmp_path / "b", {"select": "all", "max_side": "huge"}, upload)
