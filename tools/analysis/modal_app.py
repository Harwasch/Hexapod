"""Remote analysis wrapper. Image build, inference and GPU behavior are unverified.

Imported only by the Modal CLI. No deployment or provider call happens on ordinary
package import. Backends intentionally refuse inference until weights and alignment
have been validated with the camera/depth convention in hex_analysis.lifting.
"""

import base64
import io
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import modal
import numpy as np
import numpy.typing as npt
from PIL import Image

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "pipeline" if modal.is_local() else Path("/opt/pipeline")
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(HERE))

import capture_bundle  # noqa: E402

from hex_analysis.bundle import context, read_frames  # noqa: E402
from hex_analysis.registry import AnalysisUnavailableError, Registry  # noqa: E402

app = modal.App("hex-analysis")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy>=1.26", "pillow>=10", "imageio-ffmpeg>=0.5")
    .add_local_dir(str(PIPELINE), remote_path="/opt/pipeline", ignore=[".venv", "__pycache__"])
    .add_local_dir(str(HERE / "hex_analysis"), remote_path="/root/hex_analysis")
)


@app.function(image=image, cpu=4, memory=8192, timeout=600, max_containers=4, retries=0)  # type: ignore[untyped-decorator]
def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    # Only the API supplies this URL. It is an expiring read capability for one bundle;
    # tools and the language client never receive storage credentials or the URL.
    url = payload["bundleUrl"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("bundle access requires HTTPS")
    artifacts: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "capture.hexcap"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:  # noqa: S310
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > capture_bundle.MAX_BYTES:
                    raise ValueError("bundle too large")
                output.write(chunk)
        bundle = capture_bundle.unpack(archive, root / "bundle")
        frames = read_frames(bundle.directory, bundle.frames, payload["keyframes"])
        if payload["action"] == "context":
            return {"content": context(frames)}

        def write_mask(index: int, mask: npt.NDArray[np.bool_]) -> str:
            ref = f"analysis/mask-{len(artifacts)}-{index}.png"
            buffer = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255).save(buffer, format="PNG")
            artifacts.append(
                {"ref": ref, "data": base64.b64encode(buffer.getvalue()).decode("ascii")}
            )
            return ref

        try:
            entities = Registry().run(
                payload["action"], payload["arguments"], frames, payload.get("location"), write_mask
            )
        except AnalysisUnavailableError as error:
            return {"error": str(error), "entities": [], "artifacts": []}
        return {"entities": entities, "artifacts": artifacts}
