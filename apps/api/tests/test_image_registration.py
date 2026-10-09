from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile

from app.analysis.image_registration import fit
from app.analysis.image_registration_job import build
from app.schemas.land_image_registrations import ImageControlPoint, ImageRegistrationCreate
from app.services.land_image_registrations import render_image


def request(width: int = 600, height: int = 400, sha: str = "a" * 64) -> ImageRegistrationCreate:
    """Synthetic, known correspondence over the public National Mall; not an archive claim."""
    crs = CRS.from_proj4("+proj=aeqd +lat_0=38.889 +lon_0=-77.05 +datum=WGS84 +units=m")
    reverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    points = []
    for index, (x, y) in enumerate(
        [(0, 0), (width, 0), (width, height), (0, height), (width / 2, height / 2)]
    ):
        longitude, latitude = reverse.transform(2 * x - width + 0.2 * y, height - 2 * y + 0.1 * x)
        points.append(
            ImageControlPoint(
                label=f"Point {index + 1}",
                image_x=x,
                image_y=y,
                longitude=longitude,
                latitude=latitude,
            )
        )
    return ImageRegistrationCreate(
        request_key=uuid.uuid4(), name="Synthetic alignment", image_sha256=sha, points=points
    )


def test_affine_recovers_known_correspondence_and_densifies_extent() -> None:
    payload = request()
    result = fit(payload, 600, 400)
    assert result.rms_error_m < 0.01
    assert result.leave_one_out_rms_m is not None and result.leave_one_out_rms_m < 0.02
    assert result.control_point_coverage == 1
    assert len(result.footprint.coordinates[0]) == 65
    transform = Affine(*result.transform)
    reverse = Transformer.from_crs(result.crs, "EPSG:4326", always_xy=True)
    for point in payload.points:
        assert reverse.transform(*(transform * (point.image_x, point.image_y))) == pytest.approx(
            (point.longitude, point.latitude), abs=1e-7
        )
    noisy = payload.model_copy(deep=True)
    noisy.points[-1].latitude += 0.001
    noisy_result = fit(noisy, 600, 400)
    assert noisy_result.rms_error_m > 30
    assert (
        noisy_result.leave_one_out_rms_m is not None
        and noisy_result.leave_one_out_rms_m > noisy_result.rms_error_m
    )
    assert any("two source-image pixels" in warning for warning in noisy_result.warnings)


def test_rejects_degenerate_points_outside_pixels_and_unsupported_extent() -> None:
    payload = request()
    payload.points[0].image_x = 601
    with pytest.raises(ValueError, match="outside"):
        fit(payload, 600, 400)
    payload = request()
    for index, point in enumerate(payload.points):
        point.image_x = index * 10
        point.image_y = index * 10
    with pytest.raises(ValueError, match="collinear"):
        fit(payload, 600, 400)
    payload = request()
    payload.points[0].longitude = 179.9
    with pytest.raises(ValueError, match="date-line"):
        fit(payload, 600, 400)
    payload = request()
    payload.points[0].latitude += 5
    with pytest.raises(ValueError, match="250 km"):
        fit(payload, 600, 400)
    payload = request()
    payload.points[0] = payload.points[1]
    with pytest.raises(ValueError, match="distinct"):
        ImageRegistrationCreate.model_validate(payload.model_dump())


def test_render_resizes_preserving_pixel_edges_and_rgba(tmp_path: Path) -> None:
    result = fit(request(2048, 1024), 2048, 1024)
    source = tmp_path / "source.png"
    output = tmp_path / "aligned.tif"
    Image.new("RGBA", (2048, 1024), (100, 50, 20, 128)).save(source)
    assert build(source, result, output) == (1024, 512)
    with MemoryFile(output.read_bytes()) as memory, memory.open() as dataset:
        assert dataset.colorinterp == (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
        assert dataset.transform * (1024, 512) == pytest.approx(
            Affine(*result.transform) * (2048, 1024)
        )
        assert dataset.tags()["algorithm"] == "affine-aeqd-v1"
        np.testing.assert_allclose(dataset.read()[:, 200, 200], [100, 50, 20, 128], atol=1)
    # Exercise the actual bounded subprocess rather than substituting a render stub.
    stream = io.BytesIO()
    Image.new("RGBA", (600, 400), (100, 50, 20, 255)).save(stream, format="PNG")
    data, width, height = render_image(stream.getvalue(), fit(request(), 600, 400))
    assert data[:4] in (b"II*\x00", b"MM\x00*") and (width, height) == (600, 400)
