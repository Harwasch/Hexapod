"""Explicit reproducible source/weight preparation; never run by gateway startup.

Without --download-weights this downloads source only. Weight downloads are
large and must be explicitly requested by the operator. No compute is provisioned.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess

PACKAGES = {'forge-wm': 'forgewm', 'matrix-game-3': 'matrix_game', 'sana-wm': 'sana_wm'}


def patch_matrix(source):
    path = source / 'pipeline/inference_pipeline.py'
    text = path.read_text()
    original = '            exit()'
    replacement = '            return video  # worlds: preserve the resident process'
    if replacement in text:
        return
    if text.count(original) != 1:
        raise RuntimeError('Pinned Matrix source changed; refusing an unreviewed patch')
    path.write_text(text.replace(original, replacement))


def prepare_source(manifest, destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Source destination already exists; use an empty location')
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/' + manifest['repository'] + '.git', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', manifest['commit']], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != manifest['commit']:
        raise RuntimeError('Source revision mismatch')
    source = destination / manifest.get('sourceSubdirectory', '')
    if manifest['id'] == 'matrix-game-3':
        patch_matrix(source)
    (source / '.worlds-source.json').write_text(json.dumps({'commit': actual, 'patchVersion': 1}) + '\n')
    return source


def download_weights(manifest, destination):
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest['downloads']:
        snapshot_download(repo_id=item['repository'], revision=item['revision'],
            allow_patterns=item['patterns'], local_dir=str(destination / item['destination']))
    if manifest['id'] == 'forge-wm':
        base = destination / 'MG2-base'
        for name in ('diffusion_pytorch_model.safetensors', 'base_config.json'):
            shutil.copy2(base / 'base_model' / name, base / name)
    (destination / '.worlds-checkpoints.json').write_text(json.dumps(manifest['downloads'], indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=PACKAGES)
    parser.add_argument('--source', type=Path, required=True, help='Empty clone destination; Matrix SOURCE env must append /Matrix-Game-3')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--download-weights', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / PACKAGES[args.model] / 'manifest.json').read_text())
    source = prepare_source(manifest, args.source)
    if args.download_weights:
        if not args.weights:
            parser.error('--download-weights requires --weights')
        download_weights(manifest, args.weights.resolve())
    print(json.dumps({'source': str(source), 'commit': manifest['commit'], 'weightsDownloaded': args.download_weights}))


if __name__ == '__main__':
    main()
