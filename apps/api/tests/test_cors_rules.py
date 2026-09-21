"""The bucket CORS rules in infra/cors/ are checked, not just written down.

Two things are easy to lose and expensive to lose:

* ``ExposeHeaders: ["ETag"]`` on the upload rule. Without it the browser reads
  ``etag === null`` from a presigned part PUT and multipart can never complete.
* agreement between the JSON documents (aws s3api / boto3 / R2) and the XML one
  (``mc cors set``, which takes XML only). Two formats drift silently.

Nothing here asserts CORS *behaviour*: moto ignores bucket CORS rules entirely
and invents response headers, so a behavioural test would be testing the mock.
What is checked is that the documents are well formed, shape-valid to botocore,
and say the same thing.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import boto3
import pytest
from moto import mock_aws

from app.config import REPO_ROOT

if TYPE_CHECKING:
    from mypy_boto3_s3.type_defs import CORSRuleTypeDef

CORS_DIR = REPO_ROOT / "infra" / "cors"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _load(name: str) -> list[dict[str, Any]]:
    document: dict[str, Any] = json.loads((CORS_DIR / name).read_text())
    rules: list[dict[str, Any]] = document["CORSRules"]
    return rules


def _from_xml(path: Path) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(path.read_text())
    rules: list[dict[str, Any]] = []
    for rule in root.findall(f"{S3_NS}CORSRule"):

        def _all(tag: str, node: ElementTree.Element = rule) -> list[str]:
            return [(item.text or "") for item in node.findall(f"{S3_NS}{tag}")]

        parsed: dict[str, Any] = {
            "ID": _all("ID")[0],
            "AllowedOrigins": _all("AllowedOrigin"),
            "AllowedMethods": _all("AllowedMethod"),
            "AllowedHeaders": _all("AllowedHeader"),
            "ExposeHeaders": _all("ExposeHeader"),
            "MaxAgeSeconds": int(_all("MaxAgeSeconds")[0]),
        }
        rules.append(parsed)
    return rules


def test_upload_rule_exposes_etag() -> None:
    (rule,) = _load("upload.json")
    assert "ETag" in rule["ExposeHeaders"], (
        "ExposeHeaders must contain ETag or the browser reads etag === null "
        "from a presigned part PUT and multipart cannot be completed"
    )
    assert "PUT" in rule["AllowedMethods"]


def test_tiles_rule_is_read_only() -> None:
    (rule,) = _load("tiles.json")
    assert set(rule["AllowedMethods"]) == {"GET", "HEAD"}


@pytest.mark.parametrize("name", ["upload.json", "tiles.json"])
def test_rules_are_shape_valid_to_botocore(name: str) -> None:
    """botocore's client-side parameter validation accepts the document.

    This is the only mechanical check that exists before deploy. It says nothing
    about whether a given provider accepts the rule -- see the R2 note in
    docs/DEPLOYMENT.md.
    """
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="cors-shape-check")
        client.put_bucket_cors(
            Bucket="cors-shape-check",
            CORSConfiguration={"CORSRules": cast("list[CORSRuleTypeDef]", _load(name))},
        )


def test_dev_xml_matches_the_json_documents() -> None:
    """mc takes XML, everything else takes JSON. They must say the same thing."""
    assert _from_xml(CORS_DIR / "dev-minio.xml") == _load("upload.json") + _load("tiles.json")
