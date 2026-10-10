from pathlib import Path
from model_support import ModelAdapter


class Adapter(ModelAdapter):
    manifest_path = Path(__file__).with_name("manifest.json")
