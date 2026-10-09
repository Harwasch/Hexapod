"""Control-plane tests use a fake engine and never execute GPU commands."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from customization_gateway import CustomizationJobs, JobError


@pytest.fixture
def service(tmp_path):
    def atomic(path, data):
        path.write_text(json.dumps(data))

    def prepare(model, stage, source, config, output, python, devices):
        output.mkdir()
        atomic(output / 'job.json', {'model': model, 'stage': stage, 'status': 'prepared', 'activationCompatible': True})

    engine = SimpleNamespace(RECIPES={'forge-wm': ['stage3-student']}, atomic_json=atomic,
        create_job=prepare, install_checkpoint=lambda model, checkpoint, base, dest, stage: dest.mkdir(),
        select_bundle=lambda model, bundle, output: output.write_text('selected'),
        manifest=lambda model: {'prefix': 'FORGE'})
    profile = {'id': 'reviewed', 'model': 'forge-wm', 'stage': 'stage3-student',
        'source': '/private/source', 'config': '/private/config', 'python': '/private/python',
        'baseBundle': '/private/base'}
    return CustomizationJobs(tmp_path / 'jobs', [profile], engine=engine)


def complete(service):
    job = service.prepare('reviewed')
    directory, record, raw = service.record(job['id'])
    raw['status'] = 'completed'
    service.engine.atomic_json(directory / 'job.json', raw)
    (directory / 'checkpoints').mkdir()
    (directory / 'checkpoints' / 'private-dataset-name.pt').write_bytes(b'fakecheckpoint')
    return service.status(job['id'])


def test_prepare_exposes_no_paths_and_requires_explicit_execution(service):
    job = service.prepare('reviewed')
    assert job['status'] == 'prepared'
    assert '/private/' not in json.dumps(service.capabilities()) + json.dumps(job)
    with pytest.raises(JobError, match='operator enablement'):
        service.run(job['id'], True)
    with pytest.raises(JobError):
        service.prepare('/private/arbitrary-config')
    with pytest.raises(JobError):
        service.status('../escape')


def test_install_is_idempotent_and_selection_is_per_model(service):
    first, second = complete(service), complete(service)
    artifact = first['artifacts'][0]['id']
    assert first['artifacts'][0]['name'] == 'checkpoint-001.pt'
    service.install(first['id'], artifact)
    service.install(first['id'], artifact)
    service.activate(first['id'], artifact, True)
    with pytest.raises(JobError):
        service.activate(second['id'], artifact, True)
    assert service.status(first['id'])['selectedArtifactId'] == artifact
    service.install(second['id'], artifact)
    service.activate(second['id'], artifact, True)
    assert service.status(first['id'])['selectedArtifactId'] is None
    assert service.status(second['id'])['selectedArtifactId'] == artifact
    service.activate(first['id'], None, False)
    assert service.status(second['id'])['selectedArtifactId'] is None


def test_cancelled_job_cannot_restart_and_interrupted_is_reported(service):
    service.allow_training = True
    job = service.prepare('reviewed')
    directory, _, raw = service.record(job['id'])
    raw['status'] = 'cancelled'
    service.engine.atomic_json(directory / 'job.json', raw)
    with pytest.raises(JobError, match='current state'):
        service.run(job['id'], True)
    raw['status'] = 'interrupted'
    service.engine.atomic_json(directory / 'job.json', raw)
    assert service.status(job['id'])['status'] == 'interrupted'


def test_checkpoint_symlink_outside_job_is_never_exposed(service, tmp_path):
    job = complete(service)
    external = tmp_path / 'private.pt'
    external.write_bytes(b'private')
    directory = service.directory(job['id'])
    (directory / 'checkpoints' / 'escape.pt').symlink_to(external)
    assert len(service.status(job['id'])['artifacts']) == 1


def test_http_auth_rejects_non_ascii_before_reading_body(service):
    import io
    from customization_gateway import handler_for

    class Connection:
        def __init__(self):
            self.source = io.BytesIO(b'POST /jobs HTTP/1.0\r\nAuthorization: Bearer \xff\r\nContent-Length: 999999999\r\n\r\n')
            self.output = bytearray()
            self.timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def makefile(self, mode, buffering=None):
            return self.source

        def sendall(self, data):
            self.output.extend(data)

    connection = Connection()
    handler_for(service, 'x' * 24)(connection, ('127.0.0.1', 0), SimpleNamespace())
    assert connection.timeout == 15
    assert b'401 Unauthorized' in connection.output
