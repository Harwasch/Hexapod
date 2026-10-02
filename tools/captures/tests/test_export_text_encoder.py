"""`export_text_encoder`: the query side must be the same model and prompt as the image side.

The export itself needs torch and runs in the API image's build stage (it checks its own
parity there); here only what keeps the two sides in step, and the token layout.
"""

from __future__ import annotations

import export_text_encoder as ete
import segment_models as sm


def test_same_model_and_template_as_segmentation() -> None:
    assert ete.MODEL == sm.SIGLIP_MODEL
    assert ete.TEMPLATE == sm.TAG_TEMPLATE
    assert ete.TEMPLATE.format("spool") == sm.tag_prompt("Spool")
    assert ete.DIM == sm.SiglipEmbedder().dim


class _Pieces:
    def encode(self, text: str) -> list[int]:
        return [100 + ord(c) for c in text]


def test_token_ids_lower_case_eos_and_padding() -> None:
    ids = ete.token_ids(_Pieces(), "AB")
    assert ids[:3] == [100 + ord("a"), 100 + ord("b"), ete.EOS_ID]
    assert ids[3:] == [ete.PAD_ID] * (ete.MAX_LENGTH - 3)
    long = ete.token_ids(_Pieces(), "x" * 200)
    assert len(long) == ete.MAX_LENGTH
    assert long[-1] == ete.EOS_ID
