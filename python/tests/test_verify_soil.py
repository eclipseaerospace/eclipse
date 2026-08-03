# SPDX-License-Identifier: Apache-2.0
#
# Tests for tools/verify_soil.py.
#
# The verifier's only value is that it can disagree with biome, so these tests
# assert its independence structurally and then run it against the real corpus.
# Running it is paired with running it against a deliberately corrupted copy,
# because a --check that always succeeds would pass the first test alone.

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]
VERIFIER_PATH: Final = REPOSITORY_ROOT / "tools" / "verify_soil.py"
SOIL_DIRECTORY: Final = REPOSITORY_ROOT / "data" / "soils"
SOIL_PATHS: Final = sorted(SOIL_DIRECTORY.glob("*.toml"))


def _imported_modules() -> set[str]:
    tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _run_verifier(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFIER_PATH), *map(str, arguments)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def test_verifier_exists() -> None:
    assert VERIFIER_PATH.is_file(), f"missing {VERIFIER_PATH}"
    assert SOIL_PATHS, f"no soil files under {SOIL_DIRECTORY}"


def test_verifier_imports_nothing_from_biome() -> None:
    offending = sorted(
        name for name in _imported_modules() if name.split(".")[0] == "biome"
    )
    assert not offending, (
        f"verify_soil.py imports {offending} from the library it checks, which "
        "would prove only that the code agrees with itself"
    )


def test_verifier_imports_only_the_standard_library() -> None:
    offending = sorted(
        name
        for name in _imported_modules()
        if name.split(".")[0] not in sys.stdlib_module_names
    )
    assert not offending, (
        f"verify_soil.py imports {offending}; the verifier stays standard "
        "library only so its numerical path shares nothing with biome, not "
        "even numpy"
    )


def test_verifier_accepts_the_committed_corpus() -> None:
    completed = _run_verifier()
    assert completed.returncode == 0, (
        f"verify_soil.py rejects the committed soil files:\n{completed.stdout}"
        f"{completed.stderr}"
    )


@pytest.mark.parametrize("path", SOIL_PATHS, ids=lambda path: path.stem)
def test_verifier_rejects_a_perturbed_pressure(path: Path, tmp_path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "pressure_kPa = "
    assert marker in text, f"{path}: no verification cases to perturb"
    head, _, tail = text.partition(marker)
    stored, _, remainder = tail.partition(",")
    perturbed = tmp_path / path.name
    perturbed.write_text(
        f"{head}{marker}{float(stored) * (1.0 + 1e-6)},{remainder}", encoding="utf-8"
    )

    completed = _run_verifier(perturbed)
    assert completed.returncode == 1, (
        "verify_soil.py accepted a pressure perturbed by one part in a million, "
        f"so --check is not actually checking:\n{completed.stdout}"
    )
    assert "FAIL" in completed.stdout


def test_verifier_rejects_a_verified_model_it_has_no_equation_for(
    tmp_path: Path,
) -> None:
    path = SOIL_DIRECTORY / "kls1.toml"
    text = path.read_text(encoding="utf-8")
    marker = 'status     = "not_reproducible"'
    assert marker in text, f"{path}: nothing marked not_reproducible to promote"
    promoted = tmp_path / path.name
    promoted.write_text(text.replace(marker, 'status     = "verified"'), encoding="utf-8")

    completed = _run_verifier(promoted)
    assert completed.returncode == 1
    assert "no equation is typed here" in completed.stdout
