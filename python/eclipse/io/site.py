# SPDX-License-Identifier: Apache-2.0
#
# eclipse.io.site — a place, as a parameter vector.
#
# The site abstraction was declared on Day 6 and held one instance for five
# days, which is long enough for a schema to be shaped by the one case it has
# seen. It was: every field the first file carried was mandatory, because the
# first file carried all of them.
#
# The second family breaks that immediately and in a way worth recording. NASA
# states an extent, a distance from the pole and a relief for de Gerlache Rim 2
# because that region got a press release; for most of the candidate regions it
# states nothing quotable at all, and for the archived terrain products there is
# no published centre coordinate anywhere. So the region block is optional
# throughout and its absence is a value rather than a gap in the file.
#
# What is mandatory is the part a study cannot proceed without: an identity, a
# body, the crew comparison the whole legged case is argued against, and a
# declaration of which axes are populated. A site with no terrain product is
# still a site -- it is a candidate region this project cannot yet analyse, and
# saying so in a file is the difference between a coverage gap and a silence.
#
# Terrain resolves through the manifest rather than through a path, so a site
# names a product and the manifest says where the bytes come from and what they
# should hash to. Nothing here reads a filename.
#
# References
#   NASA (2024) NASA Provides Update to Artemis III Moon Landing Regions.
#   Rice JW et al. (2023) Artemis III Candidate Landing Region Geology. LPSC.

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "AXIS_NAMES",
    "CrewLimit",
    "Site",
    "load_site",
    "load_sites",
]

SCHEMA_VERSION: Final = 2

# The six axes Day 6 declared. A site file lists all of them or fails; the
# point of the vector is that an empty axis is visible rather than absent.
AXIS_NAMES: Final = (
    "entry_slope_distribution",
    "illumination_duty_cycle",
    "cold_trap_range_and_depth",
    "boulder_size_frequency",
    "annual_maximum_temperature",
    "direct_to_earth_visibility",
)

POPULATED: Final = "populated"


@dataclass(frozen=True, slots=True)
class CrewLimit:
    """What a suited crew can cross, which is what a legged platform is argued against.

    Carried on every site rather than as a global constant because it is a
    programme decision that could differ by region, and a number that quietly
    became universal would be impossible to revise later.
    """

    maximum_slope_deg: float
    traverse_range_km: float
    source: str

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_slope_deg < 90.0:
            raise ValueError(
                "crew maximum_slope_deg must lie strictly between 0 and 90 "
                f"degrees; got {self.maximum_slope_deg}"
            )
        if self.traverse_range_km <= 0.0:
            raise ValueError(
                "crew traverse_range_km must be positive; got "
                f"{self.traverse_range_km}"
            )


@dataclass(frozen=True, slots=True)
class Site:
    """One named place, with what is known about it and what is not."""

    id: str
    name: str
    body: str
    soil: str
    gravity_m_per_s2: float
    is_candidate: bool
    programme: str
    crew: CrewLimit
    terrain_product: str | None
    terrain_absence: str | None
    axes: Mapping[str, str]
    stated_extent_km: float
    stated_distance_from_pole_km: float
    stated_relief_m: float
    notes: str

    def __post_init__(self) -> None:
        if self.gravity_m_per_s2 <= 0.0:
            raise ValueError(
                f"{self.id}: gravity must be positive; got {self.gravity_m_per_s2}"
            )
        missing = [name for name in AXIS_NAMES if name not in self.axes]
        if missing:
            raise ValueError(
                f"{self.id}: the axis vector must declare every axis, populated "
                f"or not; {len(missing)} are absent, first {missing[0]!r}"
            )
        if (self.terrain_product is None) == (self.terrain_absence is None):
            raise ValueError(
                f"{self.id}: a site names a terrain product or states why it "
                "has none, and exactly one of those; got product "
                f"{self.terrain_product!r} and absence {self.terrain_absence!r}"
            )

    @property
    def has_terrain(self) -> bool:
        return self.terrain_product is not None

    @property
    def populated_axes(self) -> tuple[str, ...]:
        return tuple(name for name in AXIS_NAMES if self.axes[name] == POPULATED)


def _optional_float(table: Mapping[str, object], key: str) -> float:
    value = table.get(key)
    if value is None:
        return math.nan
    return float(value)  # type: ignore[arg-type]


def load_site(path: Path | str) -> Site:
    location = Path(path)
    table = tomllib.loads(location.read_text(encoding="utf-8"))
    version = int(table["schema_version"])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{location} declares schema_version {version}; this loader reads "
            f"{SCHEMA_VERSION}. Version 1 predated candidate regions without a "
            "terrain product and made the region block mandatory"
        )
    site = table["site"]
    region = table.get("region", {})
    terrain = table.get("terrain", {})
    return Site(
        id=str(table["id"]),
        name=str(site["name"]),
        body=str(site["body"]),
        soil=str(site["soil"]),
        gravity_m_per_s2=float(site["gravity"]["value"]),
        is_candidate=bool(region.get("candidate", False)),
        programme=str(region.get("programme", "none")),
        crew=CrewLimit(
            maximum_slope_deg=float(region["crew_limit"]["maximum_slope_deg"]),
            traverse_range_km=float(region["crew_limit"]["traverse_range_km"]),
            source=str(region["crew_limit"]["source"]),
        ),
        terrain_product=(
            str(terrain["product"]) if terrain.get("product") is not None else None
        ),
        terrain_absence=(
            str(terrain["absence"]) if terrain.get("absence") is not None else None
        ),
        axes=dict(table["axes"]),
        stated_extent_km=_optional_float(region, "extent_km"),
        stated_distance_from_pole_km=_optional_float(
            region, "distance_from_pole_km"
        ),
        stated_relief_m=_optional_float(region, "relief_m"),
        notes=str(site.get("notes", "")),
    )


def load_sites(directory: Path | str) -> dict[str, Site]:
    """Every site file in a directory, by identifier, in sorted filename order.

    Sorted so that a report generated from this is byte-identical between runs
    and between filesystems, which directory order is not.
    """
    location = Path(directory)
    sites: dict[str, Site] = {}
    for path in sorted(location.glob("*.toml")):
        site = load_site(path)
        if site.id in sites:
            raise ValueError(
                f"{location} declares the site id {site.id!r} twice; the second "
                f"is {path.name}"
            )
        sites[site.id] = site
    if not sites:
        raise ValueError(f"{location} holds no site files")
    return sites
