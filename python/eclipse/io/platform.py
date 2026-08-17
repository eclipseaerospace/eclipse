# SPDX-License-Identifier: Apache-2.0
#
# eclipse.io.platform — load a platform definition from TOML.
#
# Deliberately thinner than the soil loader. A soil file carries provenance,
# validity ranges, verification cases and recorded anomalies, because the whole
# study rests on being able to say where each number came from. A platform file
# carries assumptions, and the honest thing to record about an assumption is
# that it is one.
#
# So there is no validity range and no verification block here. Adding them
# would dress a set of invented numbers in the apparatus of measured ones, and
# that apparatus is the only reason to trust a soil file.
#
# Keys under [platform.parameters] are passed straight to the Platform
# constructor. A renamed or misspelled parameter therefore fails at
# construction rather than being silently dropped.

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from eclipse.platform import Platform

__all__ = ["PlatformDefinition", "PlatformFileError", "load_platform"]

SUPPORTED_SCHEMA_VERSION: Final = 1


class PlatformFileError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlatformDefinition:
    schema_version: int
    id: str
    name: str
    morphology: str
    basis: str
    status: str
    platform: Platform
    source_path: Path


def _require(table: dict[str, Any], key: str, context: str) -> Any:
    if key not in table:
        raise PlatformFileError(f"{context}: missing required key {key!r}")
    return table[key]


def load_platform(path: Path | str) -> PlatformDefinition:
    source_path = Path(path)
    try:
        table = tomllib.loads(source_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PlatformFileError(f"{source_path}: not valid TOML: {error}") from error

    version = _require(table, "schema_version", str(source_path))
    if version != SUPPORTED_SCHEMA_VERSION:
        raise PlatformFileError(
            f"{source_path}: schema_version {version} is not supported; this "
            f"loader reads version {SUPPORTED_SCHEMA_VERSION}"
        )

    definition = _require(table, "platform", str(source_path))
    parameters = _require(definition, "parameters", f"{source_path} [platform]")

    try:
        platform = Platform(**parameters)
    except TypeError as error:
        raise PlatformFileError(
            f"{source_path}: [platform.parameters] does not match the Platform "
            f"constructor: {error}"
        ) from error

    return PlatformDefinition(
        schema_version=version,
        id=_require(table, "id", str(source_path)),
        name=_require(definition, "name", f"{source_path} [platform]"),
        morphology=_require(definition, "morphology", f"{source_path} [platform]"),
        basis=_require(definition, "basis", f"{source_path} [platform]"),
        status=_require(definition, "status", f"{source_path} [platform]"),
        platform=platform,
        source_path=source_path,
    )
