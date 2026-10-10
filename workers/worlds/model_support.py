"""CPU-only adapter metadata and bounded isolated upstream inference protocol."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from astronex.resident_client import ResidentClient

ACTIONS = ['forward', 'backward', 'left', 'right', 'look_left', 'look_right', 'look_up', 'look_down', 'stop']


def capabilities(manifest):
    return {
        'input': {'text': manifest['text'], 'image': True, 'requiredImage': True, 'multiImage': False, 'video': False, 'audio': False},
        'control': {'wasd': True, 'mouseLook': False, 'camera6DoF': False, 'gamepad': False, 'discreteActions': True, 'continuousActions': False, 'semanticActions': False, 'promptDuringRollout': False, 'promptSwitching': manifest['text'], 'timedEvents': False, 'characterReference': False},
        'output': {'video': True, 'audio': False, 'depth': False, 'cameraPose': False},
        'runtime': {'resolutionOptions': [manifest['resolution']], 'qualityOptions': ['balanced'], 'realtime': False, 'interactionMode': 'next-clip'},
        'persistence': {'nativeMemory': False, 'snapshotRestore': False, 'deterministicSeed': True},
        'customization': {'lora': False, 'fineTune': False, 'adapters': False},
        'nativeActions': ACTIONS, 'resumeKind': 'visual-checkpoint',
    }


class ModelAdapter:
    def __init__(self):
        self.manifest = json.loads(self.manifest_path.read_text())
        self.model_id = self.manifest['id']
        self.capabilities = capabilities(self.manifest)
        prefix = self.manifest['prefix']
        self.source = Path(os.environ.get(prefix + '_SOURCE', '/opt/' + self.model_id)).resolve()
        self.weights = Path(os.environ.get(prefix + '_WEIGHTS', '/models/' + self.model_id)).resolve()
        from residency import WarmResidentPool
        # SANA refiner module caches need a separate GPU isolation audit.
        self.pool = WarmResidentPool(idle_seconds=0 if self.model_id == 'sana-wm' else None)

    def ready(self):
        try:
            marker = json.loads((self.source / '.worlds-source.json').read_text())
            if marker != {'commit': self.manifest['commit'], 'patchVersion': 1}:
                return False, 'Pinned source has not been prepared by bootstrap'
            for base, paths in ((self.source, self.manifest['sourceFiles']), (self.weights, self.manifest['weightFiles'])):
                for name in paths:
                    path = base / name
                    if not path.is_file() or not path.stat().st_size:
                        return False, 'Missing model artifact: ' + name
                    if name.endswith('.index.json'):
                        index = json.loads(path.read_text())
                        for shard in set(index.get('weight_map', {}).values()):
                            if not isinstance(shard, str) or not (path.parent / shard).resolve().is_relative_to(path.parent.resolve()) or not (path.parent / shard).is_file():
                                return False, 'Missing checkpoint shard'
                        if not index.get('weight_map'):
                            return False, 'Empty checkpoint shard index'
            return True, 'Artifacts present; CUDA inference and performance have not been validated'
        except (OSError, ValueError, TypeError):
            return False, 'Pinned source/checkpoint artifacts are incomplete'

    def metadata(self):
        ready, reason = self.ready()
        return {'id': self.model_id, 'status': 'ready' if ready else 'unavailable', 'reason': reason,
                'capabilities': self.capabilities, 'version': self.manifest['commit'],
                'checkpointRevision': self.manifest['checkpointRevision'], 'gpuInferenceVerified': False,
                'readiness': 'artifacts-verified-gpu-unverified' if ready else 'artifacts-incomplete', 'servingMode': 'resident-chunks'}

    def open_session(self, env, stop_event=None):
        prefix = self.manifest['prefix']
        command = [os.environ.get(prefix + '_PYTHON', os.environ.get('INFERENCE_PYTHON', sys.executable)),
                   str(self.manifest_path.with_name('resident.py')), '--source', str(self.source), '--weights', str(self.weights)]
        client = self.pool.acquire(lambda: ModelClient(self.source, self.weights, env, command=command, stop_event=stop_event))
        client.expected_fps = self.manifest['fps']
        return client


    def release_session(self, client, clean):
        self.pool.release(client, clean)

    def close(self):
        self.pool.close()


class ModelClient(ResidentClient):
    expected_fps = 16

    def _validate_output(self, result, directory):
        # Reuse audited path/raw-frame validation, with this model's actual fps.
        fps = result.get('fps')
        if fps != self.expected_fps:
            raise RuntimeError('Upstream model returned an unexpected frame rate')
        normalized = dict(result, fps=24)
        ResidentClient._validate_output(normalized, directory)


def write_frames(frames, directory, fps):
    """Write bounded uint8 [T,H,W,3] output without a lossy encode/decode cycle."""
    import numpy as np
    from PIL import Image
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3 or not 1 <= len(frames) <= 64:
        raise ValueError('Unexpected upstream output shape')
    if frames.dtype != np.uint8:
        raise ValueError('Upstream output must be converted explicitly to uint8')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / 'frames.rgb'
    np.ascontiguousarray(frames).tofile(raw)
    last = directory / 'continuation.png'
    Image.fromarray(frames[-1]).save(last)
    return {'rawFrames': {'path': str(raw), 'width': frames.shape[2], 'height': frames.shape[1], 'count': len(frames), 'pixelFormat': 'rgb24'},
            'continuationPath': str(last), 'fps': fps, 'nativeKVContinuity': False, 'generatedFrames': len(frames)}


def serve(engine_class):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    # Upstream libraries print arbitrary paths/prompts. Keep all logs off protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    with open(os.devnull, 'w') as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
    def emit(value):
        protocol.write(json.dumps(value, separators=(',', ':')) + '\n')
        protocol.flush()
    sys.path.insert(0, str(args.source.resolve()))
    os.chdir(args.source)
    # Every artifact is installed beforehand at immutable revisions.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    emit({'type': 'ready', 'protocolVersion': 1})
    engine = None
    load_seconds = 0.0
    for line in iter(lambda: sys.stdin.buffer.readline(65537), b''):
        try:
            if len(line) > 65536:
                raise ValueError('Request limit')
            request = json.loads(line)
            if request.get('op') == 'reset':
                if engine is not None:
                    engine.reset()
                request = None
                emit({'type': 'reset', 'sessionStateCleared': True})
                continue
            if request.get('op') != 'generate' or request.get('action') not in ACTIONS:
                raise ValueError('Unsupported request')
            if not isinstance(request.get('seed'), int) or not 0 <= request['seed'] < 2**32:
                raise ValueError('Invalid seed')
            if request.get('quality') != 'balanced':
                raise ValueError('This checkpoint supports balanced quality only')
            if not request.get('image') or not Path(request['image']).is_file():
                raise ValueError('Reference image required')
            if engine is None:
                emit({'type': 'phase', 'phase': 'loading-model'})
                started = time.perf_counter()
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA required')
                engine = engine_class(args.source, args.weights)
                torch.cuda.synchronize()
                load_seconds = time.perf_counter() - started
            emit({'type': 'phase', 'phase': 'generating'})
            started = time.perf_counter()
            result = engine.generate(request)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            result.update(generationSeconds=seconds, generatedFPS=result['generatedFrames'] / max(seconds, .001), loadSeconds=load_seconds, loadCount=1)
            emit({'type': 'result', 'result': result})
        except Exception as error:
            code = 'gpu-out-of-memory' if type(error).__name__ == 'OutOfMemoryError' else 'inference-failed'
            emit({'type': 'error', 'code': code})
            break
