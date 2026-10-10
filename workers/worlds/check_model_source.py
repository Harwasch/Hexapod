"""GPU-free check of the exact upstream symbols used by resident adapters."""
import argparse
import ast
import json
from pathlib import Path
from model_bootstrap import PACKAGES

CONTRACTS = {
    'forge-wm': {
        'inference.py': {'load_reference_frame': ['image_path', 'device', 'dtype', 'height', 'width'], 'make_action': ['action_type', 'num_raw_frames'], 'build_conditional_dict': ['pipeline', 'pixel', 'num_frames', 'mouse_cond', 'keyboard_cond'], 'infer_causal': ['pipeline', 'conditional_dict', 'num_frames', 'height', 'width', 'device', 'dtype', 'seed']},
        'utils/wan_wrapper.py': {'WanVAEWrapper.__init__': ['vae_path', 'clip_checkpoint_path', 'clip_tokenizer_path']},
    },
    'matrix-game-3': {
        'pipeline/inference_pipeline.py': {'MatrixGame3Pipeline.__init__': ['config', 'checkpoint_dir', 'args'], 'MatrixGame3Pipeline.generate': ['text', 'pil_image', 'seed', 'num_inference_steps', 'args']},
        'utils/utils.py': {'get_data': ['num_frames', 'height', 'width', 'pil_image']},
    },
    'sana-wm': {
        'inference_video_scripts/wm/inference_sana_wm.py': {'SanaWMPipeline.__init__': ['config', 'model_path', 'device', 'refiner'], 'SanaWMPipeline.generate_streaming': ['image', 'prompt', 'c2w', 'intrinsics_vec4', 'params', 'output_mode', 'decoded_chunk_callback'], 'action_string_to_c2w': ['action']},
    },
}


def verify(model, source):
    manifest = json.loads((Path(__file__).parent / PACKAGES[model] / 'manifest.json').read_text())
    if json.loads((source / '.worlds-source.json').read_text())['commit'] != manifest['commit']:
        raise ValueError('Unexpected source commit')
    count = 0
    for relative, functions in CONTRACTS[model].items():
        tree = ast.parse((source / relative).read_text())
        for qualified, required in functions.items():
            parts = qualified.split('.')
            container = tree
            for name in parts:
                matches = [node for node in container.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
                if len(matches) != 1:
                    raise ValueError('Missing upstream symbol: ' + qualified)
                container = matches[0]
            actual = {arg.arg for arg in container.args.args + container.args.kwonlyargs}
            if not set(required).issubset(actual):
                raise ValueError('Incompatible upstream signature: ' + qualified)
            count += 1
    if model == 'matrix-game-3' and 'return video  # worlds: preserve the resident process' not in (source / 'pipeline/inference_pipeline.py').read_text():
        raise ValueError('Matrix resident return patch is missing')
    return {'model': model, 'sourceCommit': manifest['commit'], 'verifiedSymbols': count, 'gpuValidated': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=CONTRACTS)
    p.add_argument('--source', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.model, a.source)))
