"""Draw the stand-in app icon: uv run --with pillow python apps/ios/scripts/make_app_icon.py <out.png>."""
import math, sys
from PIL import Image, ImageDraw
S = 1024 * 4  # supersample, then downscale for smooth edges
img = Image.new("RGB", (S, S), (24, 38, 30))  # deep field green, no alpha (App Store rule)
d = ImageDraw.Draw(img)
c = S / 2
def hexagon(r, rot=math.pi / 6):
    return [(c + r * math.cos(rot + k * math.pi / 3), c + r * math.sin(rot + k * math.pi / 3)) for k in range(6)]
# Ground band: a lighter lower half suggests land under the scan.
d.rectangle([0, S * 0.62, S, S], fill=(38, 58, 44))
# Outer hexagon outline: the hexapod / "Hex".
d.line(hexagon(S * 0.34) + hexagon(S * 0.34)[:2], fill=(236, 240, 232), width=int(S * 0.045), joint="curve")
# Three scan rays converging on a point: LiDAR seeing the land.
p = (c, c + S * 0.06)
for ang in (-0.55, 0, 0.55):
    q = (c + math.sin(ang) * S * 0.24, c - math.cos(ang) * S * 0.20)
    d.line([q, p], fill=(255, 181, 71), width=int(S * 0.018))
r = S * 0.045
d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(255, 181, 71))
img = img.resize((1024, 1024), Image.LANCZOS)
img.save(sys.argv[1], optimize=True)
