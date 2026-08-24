# SPDX-License-Identifier: Apache-2.0
#
# Tests for eclipse.io.site.
#
# Two kinds. The committed site files are read and checked against the invariants
# the schema is supposed to carry, so adding a region adds its own tests and
# nothing is hardcoded here. And the loader's refusals are exercised on written
# files, because the refusals are what stop a half-declared site from becoming a
# silently partial result.

from __future__ import annotations

from pathlib import Path

import pytest

from eclipse.io.site import AXIS_NAMES, CrewLimit, Site, load_site, load_sites

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SITE_DIRECTORY = REPOSITORY_ROOT / "configs" / "sites"

SITES = load_sites(SITE_DIRECTORY)
MINIMAL = """
schema_version = 2
id = "somewhere"

[site]
name = "Somewhere"
body = "moon"
soil = "lunar-intercrater"
gravity = { value = 1.62, units = "m/s^2" }

[region]
candidate = true

[region.crew_limit]
maximum_slope_deg = 20.0
traverse_range_km = 2.0
source = "a report"

[terrain]
absence = "no product"

[axes]
entry_slope_distribution   = "not measured"
illumination_duty_cycle    = "not measured"
cold_trap_range_and_depth  = "not measured"
boulder_size_frequency     = "not measured"
annual_maximum_temperature = "not measured"
direct_to_earth_visibility = "not measured"
"""


def write(tmp_path: Path, text: str, name: str = "site.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- what the committed files must hold, whatever they grow into


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_every_site_declares_every_axis(site: Site) -> None:
    assert set(site.axes) >= set(AXIS_NAMES)


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_every_site_names_a_product_or_states_its_absence(site: Site) -> None:
    assert (site.terrain_product is None) != (site.terrain_absence is None)


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_a_site_without_terrain_claims_no_measured_axis(site: Site) -> None:
    # The axis vector is a claim about what has been measured, and nothing can
    # have been measured at a place with no data.
    if not site.has_terrain:
        assert site.populated_axes == ()


@pytest.mark.parametrize("site", SITES.values(), ids=list(SITES))
def test_every_site_carries_a_body_and_a_soil(site: Site) -> None:
    assert site.body
    assert site.soil
    assert site.gravity_m_per_s2 > 0.0


def test_every_named_product_appears_in_the_terrain_manifest() -> None:
    from eclipse.io.terrain import load_terrain_manifest

    products = load_terrain_manifest(
        REPOSITORY_ROOT / "data" / "terrain" / "manifest.toml"
    )
    for site in SITES.values():
        if site.terrain_product is not None:
            assert site.terrain_product in products, site.id


def test_no_two_sites_share_a_terrain_product() -> None:
    # One raster analysed under two region names would manufacture a second
    # data point out of one, which is the trap Mons Mouton sets.
    used = [s.terrain_product for s in SITES.values() if s.terrain_product]
    assert len(used) == len(set(used))


def test_the_candidate_regions_number_nine() -> None:
    assert sum(1 for s in SITES.values() if s.is_candidate) == 9


# --- the loader's refusals


def test_a_site_is_read_back_whole(tmp_path: Path) -> None:
    site = load_site(write(tmp_path, MINIMAL))
    assert site.id == "somewhere"
    assert site.is_candidate
    assert not site.has_terrain
    assert site.crew.maximum_slope_deg == pytest.approx(20.0)


def test_an_older_schema_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    text = MINIMAL.replace("schema_version = 2", "schema_version = 1")
    with pytest.raises(ValueError, match="declares schema_version 1"):
        load_site(write(tmp_path, text))


def test_a_missing_axis_is_refused(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        'direct_to_earth_visibility = "not measured"\n', ""
    )
    with pytest.raises(ValueError, match="must declare every axis"):
        load_site(write(tmp_path, text))


def test_naming_both_a_product_and_an_absence_is_refused(tmp_path: Path) -> None:
    text = MINIMAL.replace(
        'absence = "no product"',
        'absence = "no product"\nproduct = "something"',
    )
    with pytest.raises(ValueError, match="exactly one of those"):
        load_site(write(tmp_path, text))


def test_naming_neither_a_product_nor_an_absence_is_refused(tmp_path: Path) -> None:
    text = MINIMAL.replace('absence = "no product"', "")
    with pytest.raises(ValueError, match="exactly one of those"):
        load_site(write(tmp_path, text))


def test_an_impossible_crew_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 90"):
        CrewLimit(maximum_slope_deg=95.0, traverse_range_km=2.0, source="x")
    with pytest.raises(ValueError, match="traverse_range_km must be positive"):
        CrewLimit(maximum_slope_deg=20.0, traverse_range_km=0.0, source="x")


def test_two_files_with_one_identity_are_refused(tmp_path: Path) -> None:
    write(tmp_path, MINIMAL, "a.toml")
    write(tmp_path, MINIMAL, "b.toml")
    with pytest.raises(ValueError, match="declares the site id 'somewhere' twice"):
        load_sites(tmp_path)


def test_an_empty_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holds no site files"):
        load_sites(tmp_path)


def test_sites_load_in_a_stable_order(tmp_path: Path) -> None:
    # Directory order is not stable between filesystems and a report generated
    # from it would not be byte-identical between machines.
    assert list(SITES) == sorted(
        (load_site(path).id for path in sorted(SITE_DIRECTORY.glob("*.toml"))),
        key=list(SITES).index,
    )
    names = [load_site(path).id for path in sorted(SITE_DIRECTORY.glob("*.toml"))]
    assert list(SITES) == names
