"""Tiny synthetic QuickTime/MP4 containers, built box by box, for the location reader.

No real video is committed: what `services/video_location.py` reads is the box tree, and
these build the parts of it an iPhone (or an Android phone) writes -- `ftyp`, an `mdat`
of filler, and a `moov` holding a `mvhd`, a `trak`, the QuickTime `meta` (`hdlr` `mdta`,
`keys`, `ilst`) and a `udta` with a `(c)xyz` atom -- with each piece optional, so a test
can say which one a clip carries. The layouts follow Apple's QuickTime File Format
reference ("Metadata", "User data atoms") and ISO/IEC 14496-12.
"""

from __future__ import annotations

import struct

ISO6709_KEY = "com.apple.quicktime.location.ISO6709"
ACCURACY_KEY = "com.apple.quicktime.location.accuracy.horizontal"
#: QuickTime well-known data types (the `data` atom's type indicator).
UTF8, FLOAT32, FLOAT64 = 1, 23, 24


def box(kind: bytes, payload: bytes = b"") -> bytes:
    assert len(kind) == 4
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def large_box(kind: bytes, payload: bytes) -> bytes:
    """A box with a 64-bit size (`size == 1`, then the size after the type)."""
    return struct.pack(">I", 1) + kind + struct.pack(">Q", 16 + len(payload)) + payload


def data_atom(value: bytes, kind: int = UTF8) -> bytes:
    return box(b"data", struct.pack(">I", kind) + struct.pack(">I", 0) + value)


def mdta_meta(items: list[tuple[str, bytes, int]], *, full_box: bool = False) -> bytes:
    """The QuickTime `meta` atom: `hdlr` (mdta), `keys`, `ilst` (1-based key indices)."""
    hdlr = box(b"hdlr", b"\0" * 8 + b"mdta" + b"\0" * 12 + b"\0")
    entries = b"".join(
        struct.pack(">I", 8 + len(key.encode())) + b"mdta" + key.encode() for key, _, _ in items
    )
    keys = box(b"keys", b"\0\0\0\0" + struct.pack(">I", len(items)) + entries)
    ilst = box(
        b"ilst",
        b"".join(
            box(struct.pack(">I", index), data_atom(value, kind))
            for index, (_, value, kind) in enumerate(items, start=1)
        ),
    )
    return box(b"meta", (b"\0\0\0\0" if full_box else b"") + hdlr + keys + ilst)


def xyz_atom(text: str) -> bytes:
    """`udta/(c)xyz`: a 16-bit length, a 16-bit language code, then the string."""
    raw = text.encode()
    return box(b"\xa9xyz", struct.pack(">HH", len(raw), 0x15C7) + raw)


def mov(
    *,
    iso6709: str | None = None,
    accuracy: str | float | None = None,
    xyz: str | None = None,
    moov_first: bool = False,
    mdat_bytes: int = 4096,
    large_mdat: bool = False,
    accuracy_type: int = UTF8,
    trak_bytes: int = 2048,
    full_box_meta: bool = False,
) -> bytes:
    """A QuickTime movie with the location atoms asked for, and none it was not."""
    ftyp = box(b"ftyp", b"qt  " + b"\0\0\0\0" + b"qt  ")
    items: list[tuple[str, bytes, int]] = [("com.apple.quicktime.make", b"Apple", UTF8)]
    if iso6709 is not None:
        items.append((ISO6709_KEY, iso6709.encode(), UTF8))
    if accuracy is not None:
        if accuracy_type == FLOAT32:
            value = struct.pack(">f", float(accuracy))
        elif accuracy_type == FLOAT64:
            value = struct.pack(">d", float(accuracy))
        else:
            value = str(accuracy).encode()
        items.append((ACCURACY_KEY, value, accuracy_type))
    children = [
        box(b"mvhd", b"\0" * 100),
        box(b"trak", b"\0" * trak_bytes),
        mdta_meta(items, full_box=full_box_meta),
    ]
    if xyz is not None:
        children.append(box(b"udta", xyz_atom(xyz)))
    moov = box(b"moov", b"".join(children))
    filler = b"\x00" * mdat_bytes
    mdat = large_box(b"mdat", filler) if large_mdat else box(b"mdat", filler)
    wide = box(b"wide")
    return ftyp + (moov + wide + mdat if moov_first else wide + mdat + moov)


def loci_atom(lat: float, lon: float, altitude: float = 0.0) -> bytes:
    """`udta/loci` (3GPP TS 26.244 8.2.3), what ffmpeg's MP4 muxer writes a location as: a
    full box, a language code, a name, a role, then longitude, latitude and altitude as
    signed 16.16 fixed point, then two more strings."""

    def fixed(value: float) -> bytes:
        return struct.pack(">i", round(value * 65536))

    return box(
        b"loci",
        b"\0\0\0\0"
        + struct.pack(">H", 0x15C7)
        + b"\0"
        + b"\0"
        + fixed(lon)
        + fixed(lat)
        + fixed(altitude)
        + b"earth\0"
        + b"\0",
    )


def mp4_with_loci(lat: float, lon: float) -> bytes:
    ftyp = box(b"ftyp", b"isom" + b"\0\0\2\0" + b"isommp41")
    moov = box(b"moov", box(b"mvhd", b"\0" * 100) + box(b"udta", loci_atom(lat, lon)))
    return ftyp + box(b"mdat", b"\0" * 512) + moov


def mp4_with_xyz(text: str) -> bytes:
    """An MP4 as an Android phone writes it: `isom` brand, location only in `udta/(c)xyz`."""
    ftyp = box(b"ftyp", b"isom" + b"\0\0\2\0" + b"isommp41")
    moov = box(b"moov", box(b"mvhd", b"\0" * 100) + box(b"udta", xyz_atom(text)))
    return ftyp + box(b"mdat", b"\0" * 512) + moov
