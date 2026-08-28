# SPDX-License-Identifier: Apache-2.0
#
# eclipse.io.terrain — read a gridded elevation product and its georeferencing.
#
# A digital elevation model is a different class of input from everything else
# this repository loads. A soil file is a transcription: every number in it was
# typed from a paper and can be checked against the paper. A DEM is a large
# binary from an archive that already stores it better than a public repository
# could, so what is committed is the manifest -- product identifier, source,
# checksum, extent -- and the bytes are fetched.
#
# The reader is deliberately narrow. It reads uncompressed, single-band,
# strip-organised float32 TIFF, which is what the LOLA products are, and raises
# on anything else rather than growing into a general TIFF library. GDAL is not
# a dependency: it would pull a large C stack into an environment that is
# otherwise numpy alone, to parse a header this file parses in eighty lines.
#
# Georeferencing is read from the file rather than assumed. The polar
# stereographic projection is implemented here because it is four lines of
# trigonometry and having it in code means it can be checked against the
# product's own stated latitude bounds, which it is.
#
# One consequence of the projection that a constant cell size hides. Polar
# stereographic is conformal but not equidistant: the point scale factor is one
# at the pole and grows away from it, so a grid of constant map spacing does not
# have constant ground spacing. Over a polar site the effect is small, and it is
# reported rather than silently absorbed into the slope.
#
# References
#   Snyder JP (1987) Map Projections: A Working Manual. USGS Professional
#     Paper 1395.
#   Adobe Systems (1992) TIFF Revision 6.0.

from __future__ import annotations

import math
import struct
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GeoRaster",
    "TerrainFileError",
    "TerrainProduct",
    "latitude_of_radius",
    "latitudes_degrees",
    "load_terrain_manifest",
    "model_to_latitude_longitude",
    "north_azimuth_degrees",
    "point_scale_factor",
    "read_float_geotiff",
]

TIFF_LITTLE_ENDIAN: Final = b"II"
TIFF_CLASSIC_MAGIC: Final = 42

TAG_IMAGE_WIDTH: Final = 256
TAG_IMAGE_LENGTH: Final = 257
TAG_BITS_PER_SAMPLE: Final = 258
TAG_COMPRESSION: Final = 259
TAG_STRIP_OFFSETS: Final = 273
TAG_SAMPLES_PER_PIXEL: Final = 277
TAG_SAMPLE_FORMAT: Final = 339
TAG_MODEL_PIXEL_SCALE: Final = 33550
TAG_MODEL_TIEPOINT: Final = 33922

COMPRESSION_NONE: Final = 1
SAMPLE_FORMAT_FLOAT: Final = 3
FLOAT32_BITS: Final = 32


class TerrainFileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GeoRaster:
    """A grid of values with the georeferencing needed to say where they are.

    Model coordinates are the projection's own metres. The origin is the outer
    corner of the first pixel, matching the GeoTIFF tiepoint convention for
    pixel-is-area rasters, so a pixel centre sits half a cell inside it.
    """

    values: NDArray[np.float64]
    origin_x_m: float
    origin_y_m: float
    cell_size_m: float
    reference_radius_m: float

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self.values.shape
        return rows, columns

    @property
    def extent_m(self) -> tuple[float, float, float, float]:
        rows, columns = self.shape
        return (
            self.origin_x_m,
            self.origin_x_m + columns * self.cell_size_m,
            self.origin_y_m - rows * self.cell_size_m,
            self.origin_y_m,
        )

    @property
    def center_model_m(self) -> tuple[float, float]:
        rows, columns = self.shape
        return (
            self.origin_x_m + columns * self.cell_size_m / 2.0,
            self.origin_y_m - rows * self.cell_size_m / 2.0,
        )

    def center_latitude_longitude(self) -> tuple[float, float]:
        x, y = self.center_model_m
        return model_to_latitude_longitude(
            x, y, reference_radius_m=self.reference_radius_m
        )

    def arc_distance_from_pole_m(self) -> float:
        latitude, _ = self.center_latitude_longitude()
        return self.reference_radius_m * math.radians(90.0 + latitude)


def latitude_of_radius(radius_m: float, *, reference_radius_m: float) -> float:
    """South polar stereographic, inverse, for a sphere with true scale at the pole."""
    return (
        math.degrees(2.0 * math.atan(radius_m / (2.0 * reference_radius_m))) - 90.0
    )


def model_to_latitude_longitude(
    x_m: float, y_m: float, *, reference_radius_m: float
) -> tuple[float, float]:
    radius = math.hypot(x_m, y_m)
    latitude = latitude_of_radius(radius, reference_radius_m=reference_radius_m)
    return latitude, math.degrees(math.atan2(x_m, y_m))


def point_scale_factor(latitude_degrees: float) -> float:
    """Map distance over ground distance, for true scale at the pole.

    One at the pole and greater away from it, so a grid of constant map spacing
    covers slightly less ground than its cell size claims, and a slope computed
    on the nominal cell is underestimated by the same fraction.
    """
    return 2.0 / (1.0 + math.sin(math.radians(abs(latitude_degrees))))


def _read_tags(raw: bytes) -> dict[int, list[float] | list[int]]:
    if raw[:2] != TIFF_LITTLE_ENDIAN:
        raise TerrainFileError(
            "only little-endian TIFF is read here; the LOLA products are 'II' "
            f"and this file begins {raw[:2]!r}"
        )
    magic, = struct.unpack("<H", raw[2:4])
    if magic != TIFF_CLASSIC_MAGIC:
        raise TerrainFileError(
            f"expected classic TIFF (magic {TIFF_CLASSIC_MAGIC}), got {magic}; "
            "BigTIFF is a different layout and is not read here"
        )
    directory_offset, = struct.unpack("<I", raw[4:8])
    count, = struct.unpack("<H", raw[directory_offset : directory_offset + 2])

    tags: dict[int, list[float] | list[int]] = {}
    for index in range(count):
        entry = directory_offset + 2 + 12 * index
        tag, kind, length = struct.unpack("<HHI", raw[entry : entry + 8])
        if kind == 3 and length == 1:
            tags[tag] = [struct.unpack("<H", raw[entry + 8 : entry + 10])[0]]
        elif kind == 4 and length == 1:
            tags[tag] = [struct.unpack("<I", raw[entry + 8 : entry + 12])[0]]
        elif kind in (4, 12):
            value_offset, = struct.unpack("<I", raw[entry + 8 : entry + 12])
            size = 4 if kind == 4 else 8
            code = "I" if kind == 4 else "d"
            block = raw[value_offset : value_offset + size * length]
            tags[tag] = list(struct.unpack("<" + code * length, block))
    return tags


def read_float_geotiff(path: Path | str) -> GeoRaster:
    """Read an uncompressed single-band float32 GeoTIFF into a georeferenced grid.

    Everything the reader relies on is asserted against the file rather than
    assumed, because a silently mis-read raster produces a plausible slope
    distribution and no error at all.
    """
    source = Path(path)
    raw = source.read_bytes()
    tags = _read_tags(raw)

    def one(tag: int, name: str) -> float:
        if tag not in tags:
            raise TerrainFileError(f"{source}: missing required TIFF tag {name}")
        return float(tags[tag][0])

    columns, rows = int(one(TAG_IMAGE_WIDTH, "ImageWidth")), int(
        one(TAG_IMAGE_LENGTH, "ImageLength")
    )
    for tag, name, expected in (
        (TAG_COMPRESSION, "Compression", COMPRESSION_NONE),
        (TAG_SAMPLE_FORMAT, "SampleFormat", SAMPLE_FORMAT_FLOAT),
        (TAG_BITS_PER_SAMPLE, "BitsPerSample", FLOAT32_BITS),
        (TAG_SAMPLES_PER_PIXEL, "SamplesPerPixel", 1),
    ):
        found = int(one(tag, name))
        if found != expected:
            raise TerrainFileError(
                f"{source}: {name} is {found}, expected {expected}. This reader "
                "handles uncompressed single-band float32 only, which is what "
                "the LOLA products are; a wider format needs a wider reader "
                "rather than a looser check"
            )

    if TAG_MODEL_PIXEL_SCALE not in tags or TAG_MODEL_TIEPOINT not in tags:
        raise TerrainFileError(
            f"{source}: no GeoTIFF ModelPixelScale or ModelTiepoint, so the grid "
            "cannot be placed on the body and a slope from it would be unlocated"
        )
    scale_x, scale_y = float(tags[TAG_MODEL_PIXEL_SCALE][0]), float(
        tags[TAG_MODEL_PIXEL_SCALE][1]
    )
    if not math.isclose(scale_x, scale_y, rel_tol=1e-12):
        raise TerrainFileError(
            f"{source}: pixels are {scale_x} by {scale_y} m and this reader "
            "assumes square cells, which every slope expression below relies on"
        )
    tiepoint = tags[TAG_MODEL_TIEPOINT]

    start = int(tags[TAG_STRIP_OFFSETS][0])
    needed = start + rows * columns * 4
    if len(raw) < needed:
        raise TerrainFileError(
            f"{source}: the header declares {rows} by {columns} float32 samples "
            f"starting at byte {start}, which needs {needed} bytes, and the file "
            f"holds {len(raw)}. It is {needed - len(raw)} bytes short, which is "
            "what a truncated download looks like; check it against the byte "
            "count in the terrain manifest"
        )
    values = np.frombuffer(
        raw, dtype="<f4", count=rows * columns, offset=start
    ).reshape(rows, columns)

    return GeoRaster(
        values=values.astype(np.float64),
        origin_x_m=float(tiepoint[3]),
        origin_y_m=float(tiepoint[4]),
        cell_size_m=scale_x,
        reference_radius_m=1737400.0,
    )


@dataclass(frozen=True, slots=True)
class TerrainProduct:
    """One archived product: where it came from and how to know it is intact.

    The grid block is optional and redundant when present -- the reader takes
    the geometry from the file header -- which is what makes it worth having:
    a declaration the reader can be checked against rather than a declaration
    the reader depends on.
    """

    id: str
    description: str
    url: str
    sha256: str
    byte_count: int
    kind: str
    quality: Mapping[str, Any]
    grid: Mapping[str, Any] | None

    @property
    def filename(self) -> str:
        return f"{self.id}.tif"


def load_terrain_manifest(path: Path | str) -> dict[str, TerrainProduct]:
    """Every product the manifest declares, by identifier.

    Nothing here reads a product id from anywhere but this file. A study that
    hardcodes one has quietly become a study about one place, which is how the
    single-site assumption survived ten days.
    """
    location = Path(path)
    table = tomllib.loads(location.read_text(encoding="utf-8"))
    products: dict[str, TerrainProduct] = {}
    for entry in table.get("product", ()):
        identifier = str(entry["id"])
        if identifier in products:
            raise ValueError(
                f"{location} declares the product id {identifier!r} twice; "
                "identifiers resolve to filenames and must be unique"
            )
        products[identifier] = TerrainProduct(
            id=identifier,
            description=str(entry["description"]),
            url=str(entry["url"]),
            sha256=str(entry["sha256"]),
            byte_count=int(entry["bytes"]),
            kind=str(entry["kind"]),
            quality=dict(entry.get("quality", {})),
            grid=dict(entry["grid"]) if "grid" in entry else None,
        )
    if not products:
        raise ValueError(f"{location} declares no terrain products")
    return products


def north_azimuth_degrees(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Where lunar north lies, in the raster frame, at each sampled cell.

    In a polar projection every meridian points a different way on the grid, so
    this cannot be a constant. For a south-polar site north is away from the
    pole, which is outward along the radius.
    """
    x = raster.origin_x_m + (columns.astype(np.float64) + 0.5) * raster.cell_size_m
    y = raster.origin_y_m - (rows.astype(np.float64) + 0.5) * raster.cell_size_m
    return np.asarray(np.degrees(np.arctan2(x, -y)) % 360.0)


def latitudes_degrees(
    raster: GeoRaster, rows: NDArray[np.int_], columns: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Latitude of each sampled cell centre."""
    return np.asarray(
        [
            model_to_latitude_longitude(
                raster.origin_x_m + (float(c) + 0.5) * raster.cell_size_m,
                raster.origin_y_m - (float(r) + 0.5) * raster.cell_size_m,
                reference_radius_m=raster.reference_radius_m,
            )[0]
            for r, c in zip(rows, columns)
        ]
    )
