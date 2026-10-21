"""`SHIPPED_RECIPE_VERSIONS` is the fallback a deployment without `tools/pipeline` stamps
runs with, so it must say what the recipe files say."""

from __future__ import annotations

from app.services import recipes


def test_the_fallback_versions_are_the_recipe_files_versions() -> None:
    # Where the recipes are readable (here: the monorepo), version_of reads the files, so
    # this compares the fallback table with the files themselves.
    assert recipes.available()
    for name, version in recipes.SHIPPED_RECIPE_VERSIONS.items():
        assert recipes.version_of(name) == version, name
    assert sorted(recipes.SHIPPED_RECIPE_VERSIONS) == recipes.known_names()
