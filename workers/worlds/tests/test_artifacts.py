"""Verify readiness before paid GPU time, using tiny local metadata fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astronex.artifacts import CHECKPOINT_FILES, SOURCE_FILES, check_artifacts
from astronex.adapter import MANIFEST


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / "source"
        self.weights = Path(self.temp.name) / "weights"
        for base, files in ((self.source, SOURCE_FILES), (self.weights, CHECKPOINT_FILES)):
            for relative in files:
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
        (self.source / ".hexapod-reviewed-source").write_text(MANIFEST["commit"])
        (self.weights / ".hexapod-checkpoint-revision").write_text(MANIFEST["checkpointRevision"])
        for base in (self.weights, self.weights / "text_encoder"):
            (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "shard.safetensors"}}))
            (base / "shard.safetensors").write_bytes(b"fixture")
    def tearDown(self):
        self.temp.cleanup()
    def test_complete_artifacts_report_only_file_readiness(self):
        self.assertEqual(check_artifacts(self.source, self.weights, MANIFEST), (True, None))
    def test_missing_text_encoder_shard_is_not_ready(self):
        (self.weights / "text_encoder/shard.safetensors").unlink()
        ready, reason = check_artifacts(self.source, self.weights, MANIFEST)
        self.assertFalse(ready)
        self.assertIn("text encoder", reason)
    def test_empty_vae_or_tokenizer_file_is_not_ready(self):
        for file in ("vae/diffusion_pytorch_model.safetensors", "tokenizer/spiece.model"):
            path = self.weights / file
            path.write_bytes(b"")
            self.assertFalse(check_artifacts(self.source, self.weights, MANIFEST)[0])
            path.write_bytes(b"fixture")
    def test_broken_indexes_fail_without_exceptions_or_path_disclosure(self):
        index = self.weights / "model.safetensors.index.json"
        for value in ([], {}, {"weight_map": []}, {"weight_map": {"x": []}}, {"weight_map": {"x": "../../private.safetensors"}}):
            index.write_text(json.dumps(value))
            ready, reason = check_artifacts(self.source, self.weights, MANIFEST)
            self.assertFalse(ready)
            self.assertNotIn("private", reason)
    def test_missing_source_module_is_not_ready(self):
        (self.source / "models/wan22_components.py").unlink()
        self.assertFalse(check_artifacts(self.source, self.weights, MANIFEST)[0])
    def test_stale_checkpoint_marker_is_not_ready(self):
        (self.weights / ".hexapod-checkpoint-revision").write_text("old")
        self.assertFalse(check_artifacts(self.source, self.weights, MANIFEST)[0])


if __name__ == "__main__":
    unittest.main()
