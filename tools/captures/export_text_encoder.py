"""Exports SigLIP 2's text tower for query-time search by meaning (SCENE_OBJECTS.md §3 step 4).

Segmentation embeds each instance's crops with `segment_models.SiglipEmbedder` (image tower);
a search query must be embedded with the *same* model's text tower and the same prompt
template, or the cosine ranking means nothing. The API does that on its own CPU
(`apps/api/app/services/text_encoder.py`) with onnxruntime and sentencepiece -- no torch in
the image -- from what this script writes:

    tower.onnx         the text transformer from token embeddings to the pooled, projected
                       output (position embeddings, 12 layers, final norm, last token, head).
                       Weights stored as float16 behind a Cast that onnxruntime folds at load,
                       so the arithmetic is float32 and the file is half the size (172 MB).
    token_table.npy    the 256000 x 768 token embedding table, float16 (393 MB). The API
                       memory-maps it and gathers the <= 64 rows a query needs, so the 786 MB
                       float32 table never sits in RAM.
    tokenizer.model    the model's own sentencepiece model (4 MB). Lower-cased text, the
                       pieces, then `<eos>` (1), truncated to 64 and padded with `<pad>` (0) to
                       64: what `AutoProcessor(..., padding="max_length", max_length=64)`
                       gives, which `SiglipEmbedder.embed_texts` uses. SigLIP pools the last
                       position, so the padding is part of the input.
    manifest.json      the model id and revision, the template, the token ids above, and the
                       parity measured by `--check`.

Why split the table out: the full text tower is 1.13 GB in float32, 283 MB as the int8
ONNX that onnx-community publishes -- and that int8 export is not the same model for
search: cosine 0.92-0.98 to the float32 tower on our queries, and top-10 overlap with the
reference ranking as low as 3/10 on the pumpkin scan (measured 2026-10-02). This split keeps
float32 arithmetic (cosine 1.000000, max score difference 1.7e-5) at 565 MB on disk.

Runs where torch is (the API image's `text-encoder` build stage, infra/api.Dockerfile):

    uv run --no-project --with torch --with transformers --with onnx --with sentencepiece \\
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
        python export_text_encoder.py OUT_DIR

It exits non-zero if the exported path disagrees with the transformers reference (cosine
below `MIN_COSINE` on any check prompt), so a bad build stops at build time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: `segment_models.SIGLIP_MODEL`, pinned to the revision this was measured on.
MODEL = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
#: `segment_models.TAG_TEMPLATE`; tests/test_export_text_encoder.py holds them equal.
TEMPLATE = "a photo of a {}."
MAX_LENGTH = 64
PAD_ID = 0
EOS_ID = 1
DIM = 768
MIN_COSINE = 0.9999
CHECK_PROMPTS = [
    TEMPLATE.format(q)
    for q in ("cable spool", "pumpkin", "log cabin", "conifer", "flagpole", "picnic table")
]


def token_ids(sp: Any, text: str) -> list[int]:
    """What the API feeds the tower (mirrors app/services/text_encoder.py `token_ids`)."""
    ids = list(sp.encode(text.lower()))[: MAX_LENGTH - 1] + [EOS_ID]
    return ids + [PAD_ID] * (MAX_LENGTH - len(ids))


def export(out: Path) -> dict[str, Any]:
    import onnx
    import torch
    from huggingface_hub import hf_hub_download
    from onnx import TensorProto, helper, numpy_helper
    from transformers import AutoModel

    out.mkdir(parents=True, exist_ok=True)
    text = AutoModel.from_pretrained(MODEL, revision=REVISION).eval().text_model

    class Tower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder, self.norm, self.head = text.encoder, text.final_layer_norm, text.head
            pos = text.embeddings.position_embedding.weight.detach().clone()
            self.register_buffer("pos", pos[None])

        def forward(self, token_embeds: torch.Tensor) -> torch.Tensor:
            h = token_embeds + self.pos[:, : token_embeds.shape[1]]
            h = self.encoder(inputs_embeds=h, attention_mask=None).last_hidden_state
            return self.head(self.norm(h)[:, -1, :])

    fp32 = out / "tower.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            Tower().eval(), (torch.zeros(1, MAX_LENGTH, DIM),), str(fp32),
            input_names=["token_embeds"], output_names=["text_embeds"],
            dynamic_axes={"token_embeds": {0: "batch"}, "text_embeds": {0: "batch"}},
            opset_version=17, dynamo=False,
        )  # fmt: skip
    table = text.embeddings.token_embedding.weight.detach().numpy()
    np.save(out / "token_table.npy", table.astype(np.float16))

    # Weight-only float16: each large float initializer is stored as float16 and cast back.
    model = onnx.load(str(fp32))
    graph = model.graph
    casts, inits = [], []
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype == np.float32 and arr.size > 4096:
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
            inits.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            inits.append(init)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.save(model, str(out / "tower.onnx"))
    fp32.unlink()
    shutil.copyfile(
        hf_hub_download(MODEL, "tokenizer.model", revision=REVISION), out / "tokenizer.model"
    )
    return {
        "format": "hexapod.text-encoder",
        "version": 1,
        "model": MODEL,
        "revision": REVISION,
        "name": f"siglip:{MODEL}",
        "template": TEMPLATE,
        "dim": DIM,
        "maxLength": MAX_LENGTH,
        "padId": PAD_ID,
        "eosId": EOS_ID,
        "files": {
            "tower": "tower.onnx",
            "table": "token_table.npy",
            "tokenizer": "tokenizer.model",
        },
    }


def check(out: Path) -> dict[str, float]:
    """Cosine between the exported path and transformers' own, as `SiglipEmbedder` runs it."""
    import onnxruntime as ort
    import sentencepiece as spm
    import torch
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).eval()
    inputs = processor(
        text=[p.lower() for p in CHECK_PROMPTS], padding="max_length", max_length=MAX_LENGTH,
        truncation=True, return_tensors="pt",
    )  # fmt: skip
    with torch.inference_mode():
        ref = model.get_text_features(**inputs)
    ref = ref if hasattr(ref, "shape") else ref.pooler_output
    ref = torch.nn.functional.normalize(ref.float(), dim=-1).numpy()

    sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
    ids = np.array([token_ids(sp, p) for p in CHECK_PROMPTS], np.int64)
    if not (ids == inputs["input_ids"].numpy()).all():
        raise SystemExit("sentencepiece ids differ from the processor's")
    table = np.load(out / "token_table.npy", mmap_mode="r")
    session = ort.InferenceSession(str(out / "tower.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(None, {"token_embeds": np.asarray(table[ids], np.float32)})[0]
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    cos = (got * ref).sum(axis=1)
    return {"minCosine": round(float(cos.min()), 7), "prompts": len(CHECK_PROMPTS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    manifest = export(args.out)
    manifest["check"] = check(args.out)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["check"]))
    if manifest["check"]["minCosine"] < MIN_COSINE:
        print(f"exported text tower disagrees with the reference (< {MIN_COSINE})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
