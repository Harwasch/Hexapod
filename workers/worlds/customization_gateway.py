"""Optional, authenticated customization control plane on an already configured GPU host.

Profiles are operator-owned JSON; requests carry profile/job/artifact IDs, never paths
or commands. This service does not allocate GPUs, deploy, or restart inference.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shlex
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

import customization

ID = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


class JobError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class CustomizationJobs:
    def __init__(self, root, profiles, *, allow_training=False, max_seconds=21600, engine=customization):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.engine = engine
        self.allow_training = allow_training
        self.max_seconds = max(60, min(int(max_seconds), 172800))
        self.lock = threading.RLock()
        self.profiles = {}
        self.running = set()
        for profile in profiles:
            if not isinstance(profile, dict) or not ID.fullmatch(str(profile.get('id', ''))):
                raise ValueError('Operator profile needs a safe ID')
            if profile.get('model') not in engine.RECIPES or profile.get('stage') not in engine.RECIPES[profile['model']]:
                raise ValueError('Profile does not match a reviewed training recipe')
            for key in ('source', 'config', 'python', 'baseBundle'):
                if not isinstance(profile.get(key), str) or not Path(profile[key]).is_absolute():
                    raise ValueError('Profile paths must be operator-supplied absolute paths')
            self.profiles[profile['id']] = dict(profile)

    def directory(self, identity):
        if not isinstance(identity, str) or not ID.fullmatch(identity):
            raise JobError('Invalid job ID')
        path = (self.root / identity).resolve()
        if path.parent != self.root or not path.is_dir():
            raise JobError('Job not found', 404)
        return path

    def record(self, identity):
        directory = self.directory(identity)
        try:
            record = json.loads((directory / 'service.json').read_text())
            job = json.loads((directory / 'job.json').read_text())
        except (OSError, ValueError):
            raise JobError('Job record is unavailable', 409) from None
        return directory, record, job

    def save(self, directory, record):
        self.engine.atomic_json(directory / 'service.json', record)

    def capabilities(self):
        return {'configured': True, 'trainingEnabled': self.allow_training,
                'maxTrainingSeconds': self.max_seconds, 'maxConcurrentJobs': 1,
                'profiles': [{'id': p['id'], 'name': str(p.get('name', p['id']))[:100],
                              'modelId': p['model'], 'stage': p['stage'], 'devices': 8,
                              'datasetLabel': str(p.get('datasetLabel', 'Operator-prepared local dataset'))[:150],
                              'activationCompatible': p['stage'] in ('stage3-student', 'self-forcing-t121')}
                             for p in self.profiles.values()],
                'message': 'Training uses existing compute and approved local datasets. No GPU is provisioned. Full staged training only; generic LoRA is not supported.'}

    def artifact_map(self, directory):
        root = (directory / 'checkpoints').resolve()
        if not root.is_dir():
            return {}
        values = {}
        for candidate in sorted(root.rglob('*.pt'))[:100]:
            resolved = candidate.resolve()
            if resolved.is_relative_to(root) and resolved.is_file() and resolved.stat().st_size:
                identity = hashlib.sha256(str(candidate.relative_to(root)).encode()).hexdigest()[:24]
                values[identity] = resolved
        return values

    def status(self, identity):
        with self.lock:
            directory, record, job = self.record(identity)
            state = record.get('status', 'prepared')
            if identity in self.running:
                state = 'cancelling' if state == 'cancelling' else 'running'
            elif state in ('running', 'cancelling'):
                state = 'unknown'
            if job.get('status') in ('completed', 'failed', 'cancelled', 'interrupted'):
                state = job['status']
            artifacts = [{'id': artifact_id, 'name': f'checkpoint-{index+1:03}.pt',
                          'size': path.stat().st_size}
                         for index, (artifact_id, path) in enumerate(self.artifact_map(directory).items())]
            return {'id': identity, 'profileId': record['profileId'], 'modelId': job['model'],
                    'stage': job['stage'], 'status': state, 'createdAt': record['createdAt'],
                    'startedAt': record.get('startedAt'), 'finishedAt': record.get('finishedAt'),
                    'maxTrainingSeconds': self.max_seconds,
                    'activationCompatible': bool(job.get('activationCompatible')),
                    'gpuValidated': bool(job.get('gpuValidated', False)),
                    'artifacts': artifacts, 'installed': record.get('installed', []),
                    'selectedArtifactId': record.get('selectedArtifactId'),
                    'message': record.get('message', '')}

    def list_jobs(self):
        result = []
        for child in self.root.iterdir():
            if child.is_dir() and (child / 'service.json').is_file():
                result.append(self.status(child.name))
        return {'jobs': sorted(result, key=lambda job: job['createdAt'], reverse=True)}

    def prepare(self, profile_id):
        with self.lock:
            profile = self.profiles.get(profile_id)
            if profile is None:
                raise JobError('Choose an installed operator profile', 422)
            identity = uuid4().hex
            directory = self.root / identity
            try:
                self.engine.create_job(profile['model'], profile['stage'], Path(profile['source']),
                                       Path(profile['config']), directory, profile['python'], 8)
            except (ValueError, OSError, KeyError):
                raise JobError('The approved recipe is not ready. Ask the operator to verify pinned sources, local datasets, configs and base checkpoints.', 422) from None
            self.save(directory, {'profileId': profile_id, 'createdAt': time.time(), 'status': 'prepared'})
            return self.status(identity)

    def run(self, identity, confirmed):
        with self.lock:
            if confirmed is not True or not self.allow_training:
                raise JobError('Training requires operator enablement and explicit compute confirmation.', 403)
            directory, record, _job = self.record(identity)
            if self.running or any(job['status'] in ('running', 'cancelling', 'unknown') for job in self.list_jobs()['jobs']):
                raise JobError('Another training process is active or needs operator reconciliation.', 409)
            if self.status(identity)['status'] != 'prepared':
                raise JobError('This job cannot be started in its current state.', 409)
            record.update(status='running', startedAt=time.time(), message='Eight-device training requested; GPU validation occurs before launch.')
            self.save(directory, record)
            self.running.add(identity)
            thread = threading.Thread(target=self._execute, args=(identity,), daemon=True)
            thread.start()
            return self.status(identity)

    def _execute(self, identity):
        directory, _, _ = self.record(identity)
        timer = threading.Timer(self.max_seconds, lambda: self._timeout(identity))
        timer.daemon = True
        timer.start()
        try:
            result = self.engine.run_job(directory, execute=True, max_seconds=self.max_seconds)
            state = result.get('status', 'failed')
            message = 'Training ended. Inspect checkpoint compatibility before installation.'
        except Exception:
            state = 'failed'
            message = 'Training failed or the GPU prerequisites were not met. Ask the operator to inspect the local training log.'
        finally:
            timer.cancel()
        with self.lock:
            _, record, _ = self.record(identity)
            record.update(status=state, finishedAt=time.time(), message=message)
            self.save(directory, record)
            self.running.discard(identity)

    def _timeout(self, identity):
        try:
            self.cancel(identity)
        except JobError:
            pass

    def cancel(self, identity):
        with self.lock:
            directory, record, _ = self.record(identity)
            if self.status(identity)['status'] not in ('running', 'cancelling', 'unknown'):
                raise JobError('There is no active training job to cancel.', 409)
            try:
                self.engine.cancel_job(directory)
            except (ValueError, OSError):
                raise JobError('Cancellation was not confirmed. Reconcile this process on the worker.', 409) from None
            record.update(status='cancelling', message='Cancellation requested. Outputs are retained locally.')
            self.save(directory, record)
            return self.status(identity)

    def install(self, identity, artifact_id):
        with self.lock:
            directory, record, job = self.record(identity)
            if self.status(identity)['status'] != 'completed' or not job.get('activationCompatible'):
                raise JobError('Only a completed, final-stage checkpoint can be installed.', 409)
            checkpoint = self.artifact_map(directory).get(artifact_id)
            if checkpoint is None:
                raise JobError('Checkpoint artifact not found.', 404)
            if artifact_id in record.get('installed', []):
                return self.status(identity)
            profile = self.profiles.get(record['profileId'])
            if profile is None:
                raise JobError('The original operator profile is no longer available.', 409)
            bundle = directory / ('bundle-' + artifact_id)
            try:
                self.engine.install_checkpoint(job['model'], checkpoint, Path(profile['baseBundle']), bundle, job['stage'])
            except (ValueError, OSError, RuntimeError, ImportError):
                raise JobError('Checkpoint installation failed validation. The inference worker was not changed.', 422) from None
            installed = record.setdefault('installed', [])
            if artifact_id not in installed:
                installed.append(artifact_id)
            self.save(directory, record)
            return self.status(identity)

    def activate(self, identity, artifact_id, enabled):
        with self.lock:
            directory, record, job = self.record(identity)
            output = self.root / ('selection-' + job['model'] + '.env')
            if enabled:
                if artifact_id not in record.get('installed', []):
                    raise JobError('Install and verify this checkpoint before selecting it.', 409)
                self.engine.select_bundle(job['model'], directory / ('bundle-' + artifact_id), output)
                record['selectedArtifactId'] = artifact_id
            else:
                profile = self.profiles.get(record['profileId'])
                if profile is None:
                    raise JobError('The original operator profile is no longer available.', 409)
                output.write_text('export ' + self.engine.manifest(job['model'])['prefix'] + '_WEIGHTS=' + shlex.quote(profile['baseBundle']) + '\n')
                record['selectedArtifactId'] = None
            for other in self.list_jobs()['jobs']:
                if other['modelId'] == job['model'] and other['id'] != identity:
                    other_directory, other_record, _ = self.record(other['id'])
                    other_record['selectedArtifactId'] = None
                    self.save(other_directory, other_record)
            record['message'] = 'Selection applies only when the operator starts a new inference worker with its selection environment file. No worker was restarted.'
            self.save(directory, record)
            return self.status(identity)


def handler_for(service, token):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(15)
            super().setup()

        def log_message(self, *_args):
            pass

        def reply(self, code, body):
            payload = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)

        def handle_request(self):
            self.connection.settimeout(15)
            if not hmac.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), ('Bearer ' + token).encode('utf-8')):
                return self.reply(401, {'detail': 'Customization gateway authentication required.'})
            try:
                parts = self.path.split('?', 1)[0].strip('/').split('/')
                body = {}
                if self.command == 'POST':
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 4096:
                        raise JobError('A bounded JSON request is required.', 413)
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise JobError('Expected a JSON object.')
                if self.command == 'GET' and parts == ['capabilities']:
                    result = service.capabilities()
                elif self.command == 'GET' and parts == ['jobs']:
                    result = service.list_jobs()
                elif self.command == 'POST' and parts == ['jobs'] and set(body) == {'profileId'}:
                    result = service.prepare(body['profileId'])
                elif len(parts) >= 2 and parts[0] == 'jobs':
                    identity = parts[1]
                    if self.command == 'GET' and len(parts) == 2:
                        result = service.status(identity)
                    elif self.command == 'POST' and len(parts) == 3 and parts[2] == 'run' and set(body) == {'confirmTraining'}:
                        result = service.run(identity, body['confirmTraining'])
                    elif self.command == 'POST' and len(parts) == 3 and parts[2] == 'cancel' and not body:
                        result = service.cancel(identity)
                    elif self.command == 'POST' and len(parts) == 3 and parts[2] in ('install', 'enable') and set(body) == {'artifactId'}:
                        result = service.install(identity, body['artifactId']) if parts[2] == 'install' else service.activate(identity, body['artifactId'], True)
                    elif self.command == 'POST' and len(parts) == 3 and parts[2] == 'disable' and not body:
                        result = service.activate(identity, None, False)
                    else:
                        raise JobError('Unknown customization operation.', 404)
                else:
                    raise JobError('Unknown customization operation.', 404)
                self.reply(200, result)
            except JobError as error:
                self.reply(error.status, {'detail': str(error)})
            except (ValueError, KeyError, TypeError):
                self.reply(422, {'detail': 'Invalid customization request.'})
            except Exception:
                self.reply(500, {'detail': 'Customization operation failed. Ask the worker operator to inspect local records.'})

        do_GET = handle_request
        do_POST = handle_request
    return Handler


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound connections before spawning a thread, including slow unauthenticated headers."""
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8792)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--profiles', type=Path, required=True)
    args = parser.parse_args()
    token = os.getenv('WORLD_CUSTOMIZATION_TOKEN', '')
    if len(token) < 24:
        raise SystemExit('Set WORLD_CUSTOMIZATION_TOKEN to a strong secret of at least 24 characters.')
    profiles = json.loads(args.profiles.read_text())
    if not isinstance(profiles, list):
        raise SystemExit('The operator profiles file must be a JSON array.')
    service = CustomizationJobs(args.root, profiles,
                                allow_training=os.getenv('WORLD_CUSTOMIZATION_ALLOW_TRAINING') == '1',
                                max_seconds=int(os.getenv('WORLD_CUSTOMIZATION_MAX_SECONDS', '21600')))
    BoundedHTTPServer((args.host, args.port), handler_for(service, token)).serve_forever()


if __name__ == '__main__':
    main()
