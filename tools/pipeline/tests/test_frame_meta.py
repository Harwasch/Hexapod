"""`frame_meta`: EXIF and DJI XMP read from a photo's original, before it is shrunk.

Every photo here is synthetic (`drone_frames.py` says how it was assembled and from which
sources); none is a real Mavic 3 Enterprise file, so these tests prove the parser reads the
fields as ExifTool's table names them, not that a real drone writes them that way.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL.TiffImagePlugin import IFDRational

import exif
import frame_meta
from drone_frames import dji_m3e_photo, dji_xmp, iphone_photo, write_jpeg


def test_dji_rtk_fields_are_read_from_attribute_xmp() -> None:
    dji = frame_meta.read_dji_xmp(dji_xmp(longtitude_typo=True))

    assert dji is not None
    assert (dji.rtk_flag, frame_meta.rtk_flag_meaning(dji.rtk_flag)) == (50, "fixed")
    assert (dji.rtk_std_lon_m, dji.rtk_std_lat_m, dji.rtk_std_hgt_m) == (0.012, 0.011, 0.02)
    assert dji.calibrated_focal_px == pytest.approx(3713.29)
    assert dji.calibrated_center_px == (2640.0, 1978.0)
    assert (dji.gimbal_pitch_deg, dji.gimbal_yaw_deg, dji.gimbal_roll_deg) == (-90.0, -12.3, 0.0)
    assert (dji.flight_pitch_deg, dji.flight_yaw_deg, dji.flight_roll_deg) == (-3.4, -11.8, 1.2)
    # ExifTool lists the misspelt `GpsLongtitude`; firmware that writes it is still read.
    assert dji.longitude == pytest.approx(-93.4259758)
    assert dji.relative_altitude_m == pytest.approx(60.1)
    assert dji.create_date == "2026-10-01T10:20:30"
    assert dji.dewarp_flag == 0
    record = dji.to_dict()
    assert record["rtk"] == {
        "flag": 50,
        "flagMeaning": "fixed",
        "stdLonM": 0.012,
        "stdLatM": 0.011,
        "stdHgtM": 0.02,
        "diffAgeS": None,
    }
    assert str(record["absoluteAltitudeDatum"]).startswith("unverified")


def test_element_form_xmp_is_read_too() -> None:
    attribute = frame_meta.read_dji_xmp(dji_xmp())
    element = frame_meta.read_dji_xmp(dji_xmp(element_form=True))

    assert element is not None
    assert element == attribute


def test_property_names_match_case_insensitively_and_only_whole() -> None:
    text = dji_xmp(extra={"RtkStdLonX": "9.9"}).replace("drone-dji:RtkFlag", "drone-dji:rtkflag")
    dji = frame_meta.read_dji_xmp(text)

    assert dji is not None
    assert dji.rtk_flag == 50
    # A longer name that starts with the one asked for is a different property.
    assert dji.rtk_std_lon_m == 0.012


def test_a_photo_without_drone_xmp_has_no_dji_record() -> None:
    assert frame_meta.read_dji_xmp(None) is None
    assert (
        frame_meta.read_dji_xmp('<x:xmpmeta><rdf:Description xmp:Rating="5"/></x:xmpmeta>') is None
    )


def test_odd_xmp_is_read_as_far_as_it_goes_and_never_raises() -> None:
    # Cut short mid-packet, a garbage value, a negative calibrated focal.
    text = dji_xmp(extra={"RtkFlag": "fixed?", "CalibratedFocalLength": "-1"})[:-40]
    dji = frame_meta.read_dji_xmp(text)

    assert dji is not None
    assert dji.rtk_flag is None
    assert dji.calibrated_focal_px is None
    assert dji.latitude is not None
    assert frame_meta.xmp_packet(b"\xff\xd8<x:xmpmeta a='b'>") == "<x:xmpmeta a='b'>"
    assert frame_meta.xmp_packet(b"\xff\xd8 nothing") is None


def test_unverified_dji_fields_are_named_as_such() -> None:
    dji = frame_meta.read_dji_xmp(dji_xmp(extra={"RtkDiffAge": "1.0"}))

    assert dji is not None
    assert dji.unverified_fields == ("RtkDiffAge",)
    assert dji.to_dict()["unverifiedFields"] == ["RtkDiffAge"]
    assert dji.rtk_diff_age_s == 1.0
    plain = frame_meta.read_dji_xmp(dji_xmp())
    assert plain is not None and plain.unverified_fields == ()


@pytest.mark.parametrize(
    ("flag", "meaning"),
    [(0, "none"), (16, "single"), (34, "float"), (40, "float"), (49, "float"), (50, "fixed"),
     (7, "unknown"), (None, None)],
)  # fmt: skip
def test_rtk_flags_are_named_by_sitemarks_codes(flag: int | None, meaning: str | None) -> None:
    assert frame_meta.rtk_flag_meaning(flag) == meaning


def test_exif_time_with_an_offset_becomes_utc() -> None:
    time = frame_meta.exif_time({0x9003: "2026:10:01 10:20:30", 0x9291: "123", 0x9011: "-05:00"})

    assert time.utc == "2026-10-01T15:20:30.123Z"
    assert time.source == "exif-offset"
    assert time.local == "2026-10-01T10:20:30.123"
    assert time.offset == "-05:00"


def test_an_offset_across_midnight_and_a_half_hour_zone_land_on_the_right_day() -> None:
    time = frame_meta.exif_time({0x9003: "2026:01:01 01:15:00", 0x9011: "+05:30"})

    assert time.utc == "2025-12-31T19:45:00.000Z"
    assert time.offset == "+05:30"


def test_exif_time_without_an_offset_uses_gps_time_or_says_it_is_local() -> None:
    gps = {29: "2026:10:01", 7: (IFDRational(15, 1), IFDRational(20, 1), IFDRational(305, 10))}

    by_gps = frame_meta.exif_time({0x9003: "2026:10:01 10:20:30"}, gps)
    assert (by_gps.utc, by_gps.source, by_gps.offset) == (
        "2026-10-01T15:20:30.500Z",
        "gps-time",
        None,
    )
    assert by_gps.local == "2026-10-01T10:20:30"

    local = frame_meta.exif_time({0x9003: "2026:10:01 10:20:30"})
    assert (local.utc, local.source, local.local) == (None, "exif-local", "2026-10-01T10:20:30")

    # An offset field a camera filled with blanks is no offset.
    blank = frame_meta.exif_time({0x9003: "2026:10:01 10:20:30", 0x9011: "   :  "})
    assert blank.source == "exif-local"
    # A GPS time out of range is no time.
    bad = {29: "2026:10:01", 7: (IFDRational(25, 1), IFDRational(0, 1), IFDRational(0, 1))}
    assert frame_meta.exif_time({}, bad).source is None


def test_the_xmp_create_date_is_the_last_resort_and_keeps_its_zone_if_it_has_one() -> None:
    naive = frame_meta.exif_time({}, xmp_create_date="2026-10-01T10:20:30")
    assert (naive.source, naive.utc, naive.local) == (
        "xmp-createdate-local",
        None,
        "2026-10-01T10:20:30",
    )
    zoned = frame_meta.exif_time({}, xmp_create_date="2026-10-01T10:20:30+02:00")
    assert (zoned.source, zoned.utc, zoned.offset) == (
        "xmp-createdate",
        "2026-10-01T08:20:30.000Z",
        "+02:00",
    )
    assert frame_meta.exif_time({}, xmp_create_date="last tuesday").source is None


def test_the_horizontal_gps_error_is_read(tmp_path: Path) -> None:
    photo = iphone_photo(tmp_path / "IMG_0001.JPG", 0, h_error=0.03)

    fix = exif.read_fix(photo)
    assert fix is not None and fix.h_error_m == pytest.approx(0.03)
    record = frame_meta.read_photo(
        photo, frame_name="frame_0000.jpg", upload_root=tmp_path, frame_size=(1600, 1200)
    ).to_dict()
    gps = record["gps"]
    assert isinstance(gps, dict)
    assert gps["hAccuracyM"] == pytest.approx(0.03)
    # The height-datum vocabulary of the site frame, with the words beside it.
    assert gps["altitudeDatum"] == "msl-unverified"
    assert "geoid" in str(gps["altitudeNote"])


def test_an_iphone_photo_is_described_in_full(tmp_path: Path) -> None:
    photo = iphone_photo(tmp_path / "trip" / "IMG_0001.JPG", 3)

    record = frame_meta.read_photo(
        photo, frame_name="frame_0002.jpg", upload_root=tmp_path, frame_size=(800, 600)
    )
    document = record.to_dict()

    assert document["name"] == "frame_0002.jpg"
    assert (document["original"], document["originalPath"]) == ("IMG_0001.JPG", "trip/IMG_0001.JPG")
    assert document["originalSize"] == [1600, 1200]
    assert document["camera"] == {
        "make": "Apple",
        "model": "iPhone 15 Pro",
        "lens": "iPhone 15 Pro back triple camera 6.765mm f/1.78",
    }
    assert document["time"] == {
        "local": "2025-06-14T10:41:03.123",
        "offset": "+01:00",
        "utc": "2025-06-14T09:41:03.123Z",
        "source": "exif-offset",
    }
    focal = document["focal"]
    assert isinstance(focal, dict)
    assert (focal["mm"], focal["mm35"], focal["pxFrom"]) == (6.765, 24.0, "exif-35mm")
    assert focal["px"] == pytest.approx(frame_meta.focal_px_from_35mm(24, 800, 600), abs=1e-4)
    assert document["dji"] is None
    assert document["mask"] == "frame_0002.jpg.png"
    exposure = document["exposure"]
    assert isinstance(exposure, dict)
    assert (exposure["exposureTimeS"], exposure["fNumber"], exposure["iso"]) == (
        pytest.approx(1 / 120),
        1.78,
        64.0,
    )


def test_serial_numbers_are_never_read(tmp_path: Path) -> None:
    photo = write_jpeg(
        tmp_path / "a.jpg",
        size=(64, 48),
        exif={0x010F: "DJI"},
        exif_ifd={0xA431: "SERIAL-1234"},
        xmp=dji_xmp(extra={"CameraSerialNumber": "SERIAL-5678", "DroneSerialNumber": "X"}),
    )

    record = frame_meta.read_photo(
        photo, frame_name="frame_0000.jpg", upload_root=tmp_path, frame_size=(64, 48)
    )

    assert "SERIAL" not in str(record.to_dict())


def test_the_35mm_focal_is_converted_by_the_diagonal() -> None:
    assert frame_meta.focal_px_from_35mm(24, 640, 480) == pytest.approx(443.7254, abs=1e-3)
    # Not "long side over 36", which is what the subsystem map said.
    assert frame_meta.focal_px_from_35mm(24, 640, 480) != pytest.approx(24 * 640 / 36, abs=1)


def _pycolmap_python() -> str | None:
    import colmap4

    interpreter = colmap4.python()
    return interpreter if colmap4.version(interpreter) is not None else None


@pytest.mark.skipif(_pycolmap_python() is None, reason="needs pycolmap (CI installs the wheel)")
def test_the_35mm_rule_is_colmaps_own(tmp_path: Path) -> None:
    """COLMAP's own ImageReader, given a JPEG with only `FocalLengthIn35mmFilm`, seeds the
    camera with the focal `focal_px_from_35mm` computes -- the diagonal rule."""
    images = tmp_path / "images"
    write_jpeg(images / "a.jpg", size=(640, 480), exif_ifd={0xA405: 24})
    script = (
        "import sys, pycolmap\n"
        "db, images = sys.argv[1], sys.argv[2]\n"
        "pycolmap.extract_features(db, images)\n"
        "with pycolmap.Database.open(db) as d:\n"
        "    camera = d.read_all_cameras()[0]\n"
        "print(camera.params[0], camera.has_prior_focal_length)\n"
    )
    interpreter = _pycolmap_python()
    assert interpreter is not None
    out = subprocess.run(
        [interpreter, "-c", script, str(tmp_path / "db.db"), str(images)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[-1] == "True"
    assert float(out[-2]) == pytest.approx(frame_meta.focal_px_from_35mm(24, 640, 480), abs=1e-3)


def test_dji_calibrated_focal_is_scaled_to_the_frame() -> None:
    assert frame_meta.calibrated_focal_px(3713.29, (5280, 3956), (2400, 1798)) == pytest.approx(
        1687.86, abs=0.01
    )


def test_the_calibrated_focal_wins_over_the_35mm_one(tmp_path: Path) -> None:
    photo = dji_m3e_photo(tmp_path / "DJI_0001.JPG", 0, size=(3000, 2250))

    record = frame_meta.read_photo(
        photo, frame_name="frame_0000.jpg", upload_root=tmp_path, frame_size=(1600, 1200)
    )

    assert record.focal.px_from == "dji-calibrated"
    # The helper writes the M3E's 3713.29 px at 5280 scaled to the synthetic 3000 px.
    assert record.focal.px == pytest.approx(3713.29 * 3000 / 5280 * 1600 / 3000, abs=1e-6)
    assert record.time.source == "exif-local"
    assert record.dji is not None and record.dji.rtk_flag == 50


def test_ev100_of_a_known_exposure() -> None:
    assert frame_meta.ev100(1 / 1000, 2.8, 100) == pytest.approx(12.937, abs=1e-3)
    assert frame_meta.ev100(1 / 1000, 2.8, 400) == pytest.approx(10.937, abs=1e-3)
    assert frame_meta.ev100(None, 2.8, 100) is None
    assert frame_meta.ev100(0.0, 2.8, 100) is None


def _record(px: float | None, source: str | None = "exif-35mm") -> frame_meta.FrameRecord:
    return frame_meta.FrameRecord(
        name="f.jpg",
        original="o.jpg",
        focal=frame_meta.Focal(px=px, px_from=source),  # type: ignore[arg-type]
    )


def test_mixed_focal_lengths_do_not_seed_one_camera() -> None:
    f24, f26 = (frame_meta.focal_px_from_35mm(f, 1600, 1200) for f in (24, 26))
    mixed = frame_meta.capture_focal([_record(f24), _record(f26), _record(f24)])

    assert mixed["px"] is None
    assert "mixed" in str(mixed["note"])
    one = frame_meta.capture_focal([_record(f24), _record(f24), _record(None, None)])
    assert one["px"] == pytest.approx(f24, abs=1e-4)
    assert (one["frames"], one["of"], one["from"]) == (2, 3, "exif-35mm")
    sources = frame_meta.capture_focal([_record(f24), _record(f24, "dji-calibrated")])
    assert sources["px"] is None and sources["from"] == "mixed"
    assert frame_meta.capture_focal([_record(None, None)])["px"] is None


def test_an_unreadable_original_is_a_thin_record(tmp_path: Path) -> None:
    junk = tmp_path / "broken.jpg"
    junk.write_bytes(b"\xff\xd8 this is not a jpeg at all")

    record = frame_meta.read_photo(
        junk, frame_name="frame_0001.jpg", upload_root=tmp_path, frame_size=(10, 10)
    )

    assert (record.name, record.original, record.mask) == (
        "frame_0001.jpg",
        "broken.jpg",
        "frame_0001.jpg.png",
    )
    assert record.unreadable is not None
    assert record.dji is None and record.gps is None and record.original_size is None


def test_xmp_without_the_xmpmeta_wrapper_is_still_read(tmp_path: Path) -> None:
    """Some writers put a bare `rdf:RDF` in the XMP segment; Pillow surfaces it as is."""
    bare = dji_xmp().replace('<x:xmpmeta xmlns:x="adobe:ns:meta/">', "").replace("</x:xmpmeta>", "")
    photo = write_jpeg(tmp_path / "a.jpg", size=(64, 48), xmp=bare)

    record = frame_meta.read_photo(
        photo, frame_name="frame_0000.jpg", upload_root=tmp_path, frame_size=(64, 48)
    )

    assert record.dji is not None and record.dji.rtk_flag == 50
