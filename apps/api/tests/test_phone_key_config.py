"""Rotation keeps existing phone keys valid, without changing human normalization."""

import pytest

from app.config import Settings
from app.services import phone_key
from app.services.errors import UnauthorizedError


def test_multiple_phone_hashes() -> None:
    hashes = [phone_key.hash_key(key, salt=key.encode(), iterations=1000) for key in ("old", "new")]
    settings = Settings(api_phone_key_hash=", ".join(hashes))
    for key in (" OLD ", "new"):
        phone_key.check(settings, key)
    for key in ("", "wrong"):
        with pytest.raises(UnauthorizedError, match="That phone key is not right"):
            phone_key.check(settings, key)


def test_app_key_hash_is_its_own_setting_and_adds_to_the_general_one() -> None:
    general = phone_key.hash_key("general", salt=b"general", iterations=1000)
    app = phone_key.hash_key("app", salt=b"app", iterations=1000)
    only_app = Settings(api_phone_key_hash=None, api_phone_app_key_hash=app)
    assert phone_key.enabled(only_app)
    phone_key.check(only_app, "app")
    both = Settings(api_phone_key_hash=general, api_phone_app_key_hash=app)
    for key in ("general", "app"):
        phone_key.check(both, key)
    with pytest.raises(UnauthorizedError, match="That phone key is not right"):
        phone_key.check(both, "wrong")
    assert not phone_key.enabled(Settings(api_phone_key_hash=None, api_phone_app_key_hash=None))


@pytest.mark.parametrize("stored", ["bad", "pbkdf2_sha256$0$aa$bb", "pbkdf2_sha256$x$aa$bb"])
def test_malformed_phone_config(stored: str) -> None:
    with pytest.raises(UnauthorizedError, match="misconfigured"):
        phone_key.check(Settings(api_phone_key_hash=stored), "key")


def test_stdin_helper_prints_only_a_hash() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "app.scripts.phone_key_hash"],
        input="  Mixed-Case-Key\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    assert result.stdout.startswith("pbkdf2_sha256$200000$")
    assert "Mixed-Case-Key" not in result.stdout
    phone_key.check(Settings(api_phone_key_hash=result.stdout.strip()), "mixed-case-key")
