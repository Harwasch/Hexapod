"""LTX 2.5 latent-carry exploration with honest chunk-boundary control metadata."""
import json
import os
import sys
from pathlib import Path

from astronex.resident_client import ResidentClient
from residency import WarmResidentPool

from .controls import ACTIONS, ExplorationControls

CAPABILITIES = {
    'input': {'text': True, 'image': True, 'requiredImage': False, 'multiImage': False, 'video': False, 'audio': False},
    'control': {'wasd': True, 'mouseLook': True, 'camera6DoF': False, 'gamepad': True, 'discreteActions': True, 'continuousActions': True, 'semanticActions': False, 'promptDuringRollout': False, 'promptSwitching': True, 'timedEvents': False, 'characterReference': False, 'controlMode': 'prompt-adapted'},
    'output': {'video': True, 'audio': True, 'depth': False, 'cameraPose': False},
    'runtime': {'resolutionOptions': ['768x448'], 'qualityOptions': ['low-latency', 'balanced', 'quality'], 'realtime': False, 'interactionMode': 'continuous-chunks', 'controlMode': 'prompt-adapted'},
    'persistence': {'nativeMemory': False, 'snapshotRestore': False, 'deterministicSeed': True},
    'customization': {'lora': False, 'fineTune': False, 'adapters': False},
    'nativeActions': ACTIONS, 'resumeKind': 'visual-checkpoint',
}


def validate_audio(raw, directory):
    if not isinstance(raw, dict) or raw.get('sampleRate') != 48000 or raw.get('channels') != 2 or raw.get('sampleFormat') != 's16le':
        raise RuntimeError('The resident worker returned unsupported audio metadata')
    samples = raw.get('samples')
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 1440000 or not isinstance(raw.get('path'), str):
        raise RuntimeError('The resident worker returned unbounded audio')
    path = Path(raw['path'])
    if not path.is_file() or not path.resolve().is_relative_to(Path(directory).resolve()) or path.stat().st_size != samples * 4:
        raise RuntimeError('The resident worker returned audio outside its private directory or with an invalid length')
    return path


class LTXClient(ResidentClient):
    request_limit = 128 * 1024

    @staticmethod
    def _validate_output(result, directory):
        ResidentClient._validate_output(result, directory)
        validate_audio(result.get('audio'), directory)
        revision = result.get('appliedRevision')
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise RuntimeError('The resident worker returned an invalid prompt revision')


class Adapter:
    model_id = 'ltx-2.5'
    capabilities = CAPABILITIES
    continuous_chunks = True

    def __init__(self):
        self.manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
        self.source = Path(os.environ.get('LTX25_SOURCE', '/opt/ltx25')).resolve()
        self.weights = Path(os.environ.get('LTX25_WEIGHTS', '/models/ltx-2.5')).resolve()
        self.pool = WarmResidentPool(idle_seconds=0)

    def ready(self):
        from .bootstrap import check_artifacts
        report = check_artifacts(self.source, self.weights, self.manifest)
        return report['ready'], 'Artifacts present; GPU inference remains unverified' if report['ready'] else 'Prepare pinned LTX 2.5 source and checkpoint artifacts'

    def metadata(self):
        ready, reason = self.ready()
        return {'id': self.model_id, 'status': 'ready' if ready else 'unavailable', 'reason': reason,
                'capabilities': self.capabilities, 'version': self.manifest['commit'],
                'checkpointRevision': self.manifest['checkpointRevision'], 'gpuInferenceVerified': False,
                'servingMode': 'continuous-chunks', 'continuation': 'video-and-audio-latent-overlap',
                'readiness': 'artifacts-verified-gpu-unverified' if ready else 'artifacts-incomplete',
                'warmIdleSeconds': self.pool.idle_seconds, 'warmReuse': 'disabled'}

    def _controls(self, session):
        if not hasattr(session, 'exploration_controls'):
            session.exploration_controls = ExplorationControls()
            session.exploration_controls.settings['cadence'] = {'low-latency': 'responsive', 'balanced': 'balanced', 'quality': 'smooth'}[session.quality]
        return session.exploration_controls

    def apply_native(self, session, action, values):
        return self._controls(session).update(action, values)

    def control_state(self, session):
        return {**self._controls(session).snapshot(session.queued_revision),
                'basePrompt': session.base_prompt, 'promptAmendments': list(session.prompt_amendments)}

    def pause_controls(self, session):
        return self._controls(session).pause()

    def resume_controls(self, session):
        return self._controls(session).resume()

    def clear_controls(self, session):
        self._controls(session).clear()

    def open_session(self, env, stop_event=None):
        residency = os.environ.get('LTX25_RESIDENCY', 'gpu')
        if residency not in ('gpu', 'cpu'):
            raise ValueError('LTX25_RESIDENCY must be gpu or cpu')
        command = [os.environ.get('LTX25_PYTHON', os.environ.get('INFERENCE_PYTHON', sys.executable)),
                   str(Path(__file__).with_name('resident.py')), '--source', str(self.source), '--weights', str(self.weights), '--residency', residency]
        return self.pool.acquire(lambda: LTXClient(self.source, self.weights, env, command=command, stop_event=stop_event))

    def release_session(self, resident, clean):
        self.pool.release(resident, clean)

    def close(self):
        self.pool.close()
