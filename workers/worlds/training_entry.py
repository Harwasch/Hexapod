"""Run a reviewed SANA training script with a pinned local Gemma2 encoder.

Upstream's symbolic `gemma-2-2b-it` maps to a Hub ID, while our immutable
checkpoint bundle is installed with local_dir. Resolve that one known component
locally before importing any trainer. No new model/training API is invented.
"""
import argparse
from pathlib import Path
import runpy
import sys


def install_local_encoder(builder, path):
    original = builder.get_tokenizer_and_text_encoder
    def encoder(name='T5', device='cuda'):
        if name != 'gemma-2-2b-it':
            raise ValueError('This reviewed SANA training recipe requires gemma-2-2b-it')
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        tokenizer.padding_side = 'right'
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16, local_files_only=True)
        return tokenizer, model.get_decoder().to(device)
    builder.get_tokenizer_and_text_encoder = encoder
    return original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--script', required=True, choices=['train_video_scripts/train_sana_wm_stage1.py', 'train_video_scripts/train_longsana.py'])
    parser.add_argument('--gemma2', required=True, type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import diffusion.model.builder as builder
    install_local_encoder(builder, args.gemma2.resolve())
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    script = source / args.script
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__': main()
