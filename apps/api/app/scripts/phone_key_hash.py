"""Read one phone key from stdin and print only its normalized, salted PBKDF2 hash."""

import secrets
import sys

from app.services.phone_key import hash_key


def main() -> None:
    key = sys.stdin.read().strip().lower()
    if not key:
        raise SystemExit("phone key must not be empty")
    sys.stdout.write(hash_key(key, salt=secrets.token_bytes(16)) + "\n")


if __name__ == "__main__":
    main()
