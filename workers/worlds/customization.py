"""Local, explicit upstream full-training jobs and checkpoint bundle management.

No job starts without `run --run`. No cloud API is called. Stage completion is
recorded; re-running a completed job is a no-op. Failed jobs retain local logs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import subprocess
import sys
import time
from model_bootstrap import PACKAGES

ROOT = Path(__file__).resolve().parent
RECIPES = {
    'forge-wm': {
        'stage0-sft': ('train.py', 'configs/stage0_bid_sft.yaml'),
        'stage1-causal': ('train.py', 'configs/stage1_teacher_forcing.yaml'),
        'stage2-distill': ('train.py', 'configs/stage2_consistency_distillation.yaml'),
        'stage3-student': ('train.py', 'configs/stage3_dmd.yaml'),
    },
    'sana-wm': {
        'stage1-sft': ('train_video_scripts/train_sana_wm_stage1.py', 'configs/sana_wm/stage1/sana_wm_stage1_sekai_chunk_causal_cp2_fsdp2.yaml'),
        'ode': ('train_video_scripts/train_longsana.py', 'configs/sana_wm/distill/ode_t43.yaml'),
        'self-forcing-t43': ('train_video_scripts/train_longsana.py', 'configs/sana_wm/distill/self_forcing_t43.yaml'),
        'self-forcing-t121': ('train_video_scripts/train_longsana.py', 'configs/sana_wm/distill/self_forcing_t121.yaml'),
    },
}


def manifest(model):
    return json.loads((ROOT / PACKAGES[model] / 'manifest.json').read_text())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.' + str(threading.get_ident()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_source(model, source):
    marker = json.loads((source / '.worlds-source.json').read_text())
    if marker.get('commit') != manifest(model)['commit']:
        raise ValueError('Training requires the reviewed pinned source revision')


def create_job(model, stage, source, config, output, python, devices):
    if model not in RECIPES or stage not in RECIPES[model]:
        raise ValueError('No verified upstream training recipe for this model/stage')
    if devices != 8:
        raise ValueError('Published recipes require 8 devices; custom distributed recipes are not validated')
    source, config, output = source.resolve(), config.resolve(), output.resolve()
    validate_source(model, source)
    script, reference = RECIPES[model][stage]
    if not (source / script).is_file() or not config.is_file():
        raise ValueError('Training source/config is missing')
    # The operator supplies an explicit reviewed config with local dataset and
    # checkpoint paths. Never silently download a 235GB training dataset.
    import yaml
    config_data = yaml.safe_load(config.read_text())
    if not isinstance(config_data, dict):
        raise ValueError('Config must be a mapping')
    components = config_data.pop('worlds_components', {})
    if model == 'forge-wm':
        data_path = Path(str(config_data.get('data_path', '')))
        if not data_path.is_absolute() or not data_path.is_dir():
            raise ValueError('Set data_path to your absolute encoded LMDB dataset directory')
        model_path = Path(str(config_data.get('model_kwargs', {}).get('model_name', '')))
        if not model_path.is_absolute() or not model_path.is_dir():
            raise ValueError('Set model_kwargs.model_name to the absolute MG2-base directory')
        # Upstream training VAE defaults to this relative location.
        if not (source / 'ckpts/MG2-base/Wan2.1_VAE.pth').is_file():
            raise ValueError('Pinned Forge training expects ckpts/MG2-base in its source directory (link your installed base bundle)')
        for field in ('generator_ckpt', 'teacher_ckpt'):
            if config_data.get(field) and not Path(config_data[field]).is_file():
                raise ValueError('Set ' + field + ' to the preceding stage checkpoint')
    else:
        gemma = Path(str(components.get('gemma2', '')))
        if not gemma.is_absolute() or not (gemma / 'config.json').is_file() or not (gemma / 'tokenizer.json').is_file():
            raise ValueError('Set worlds_components.gemma2 to your pinned local Gemma2 bundle directory')
        data = config_data.get('data', {})
        if data.get('hf_dataset_repo'):
            raise ValueError('Disable automatic hf_dataset_repo downloads; provide your prepared local dataset paths')
        if 'hf://' in json.dumps(config_data):
            raise ValueError('Replace remote model references with pinned local checkpoints')
        paths = [config_data['data_path']] if config_data.get('data_path') else list(data.get('data_dir', {}).values())
        if not paths or any(not Path(path).is_absolute() or not Path(path).exists() for path in paths):
            raise ValueError('Set data_path or data.data_dir to absolute prepared trajectory/latent dataset paths')
        for field in ('model_path', 'fake_model_path', 'real_model_path'):
            if config_data.get(field) and not Path(config_data[field]).is_file():
                raise ValueError('Set ' + field + ' to a preceding-stage local checkpoint')
        config_data['work_dir'] = str(output / 'checkpoints')
        if stage == 'stage1-sft':
            config_data['report_to'] = 'none'
    output.mkdir(parents=True, exist_ok=False)
    copied = output / 'config.yaml'
    copied.write_text(yaml.safe_dump(config_data, sort_keys=False))
    entry = [str(source / script)]
    if model == 'sana-wm':
        entry = [str(ROOT / 'training_entry.py'), '--source', str(source), '--script', script,
                 '--gemma2', str(Path(components['gemma2']).resolve()), '--']
    argv = [str(Path(python).resolve()), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', *entry, '--config_path', str(copied)]
    if stage != 'stage1-sft':
        argv += ['--logdir', str(output / 'checkpoints'), '--disable-wandb']
    record = {'version': 1, 'model': model, 'stage': stage, 'sourceCommit': manifest(model)['commit'],
              'source': str(source), 'referenceConfig': reference, 'argv': argv, 'status': 'prepared',
              'configSha256': hashlib.sha256(copied.read_bytes()).hexdigest(), 'gpuValidated': False,
              'activationCompatible': stage in ('stage3-student', 'self-forcing-t121')}
    atomic_json(output / 'job.json', record)
    return record


def training_environment(environ=None):
    """Do not lend provider or gateway credentials to third-party training code."""
    source = os.environ if environ is None else environ
    allowed = {'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'LD_LIBRARY_PATH', 'LIBRARY_PATH',
               'CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'CUDA_PATH', 'NVIDIA_VISIBLE_DEVICES',
               'NVIDIA_DRIVER_CAPABILITIES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'XDG_CACHE_HOME', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME',
               'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'PYTORCH_ALLOC_CONF',
               'PYTORCH_CUDA_ALLOC_CONF', 'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE',
               'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE'}
    env = {key: value for key, value in source.items() if key in allowed}
    env.update(WANDB_MODE='disabled', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', PYTHONUNBUFFERED='1')
    return env


def run_job(directory, execute=False, max_seconds=21600):
    directory = directory.resolve()
    path = directory / 'job.json'
    job = json.loads(path.read_text())
    validate_source(job['model'], Path(job['source']))
    if hashlib.sha256((directory / 'config.yaml').read_bytes()).hexdigest() != job['configSha256']:
        raise ValueError('Config changed after preparation; prepare a new reviewed job')
    if not execute:
        return dict(job, executionRequested=False)
    if job['status'] == 'completed' or (directory / '.cancel-requested').exists():
        return job
    # CPU guard happens before starting a distributed process; no provisioning.
    env = training_environment()
    subprocess.run([job['argv'][0], '-c', 'import torch; assert torch.cuda.device_count() >= 8, "This pinned recipe requires eight visible CUDA GPUs"'], check=True, timeout=60, env=env)
    if (directory / '.cancel-requested').exists():
        return json.loads(path.read_text())
    lock = directory / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    job.update(status='running', startedAt=time.time())
    atomic_json(path, job)
    old_term = None
    if threading.current_thread() is threading.main_thread():
        old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        with (directory / 'training.log').open('ab') as log:
            # A separate bounded supervisor survives abrupt service death long
            # enough to kill the complete distributed training group.
            supervisor = [sys.executable, str(ROOT / 'training_supervisor.py'),
                '--parent-pid', str(os.getpid()), '--parent-start', str(_process_start(os.getpid())),
                '--max-seconds', str(min(max(int(max_seconds), 1), 172800)), '--', *job['argv']]
            process = subprocess.Popen(supervisor, cwd=job['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(pid=process.pid, processStartTime=_process_start(process.pid), deadlineAt=time.time() + min(max(int(max_seconds), 1), 172800))
            atomic_json(path, job)
            while True:
                if (directory / '.cancel-requested').exists() and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                try:
                    code = process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        cancelled = (directory / '.cancel-requested').exists()
        job.update(status='cancelled' if cancelled else ('completed' if code == 0 else 'failed'), exitCode=code, finishedAt=time.time())
        atomic_json(path, job)
        return job
    except BaseException:
        if 'process' in locals() and process.poll() is None:
            cancel_job(directory)
        job.update(status='interrupted', finishedAt=time.time())
        atomic_json(path, job)
        raise
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
        lock.unlink(missing_ok=True)


def _process_start(pid):
    # Linux start ticks distinguish our process from a recycled PID.
    try:
        return Path('/proc/' + str(int(pid)) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def cancel_job(directory, timeout=10):
    directory = Path(directory).resolve()
    path = directory / 'job.json'
    job = json.loads(path.read_text())
    if job.get('status') in ('completed', 'failed', 'cancelled', 'interrupted'):
        return job
    (directory / '.cancel-requested').touch(exist_ok=True)
    pid = job.get('pid')
    job.update(cancelRequested=True, status='cancelling' if pid else 'cancelled')
    atomic_json(path, job)
    if pid and job.get('processStartTime') and _process_start(pid) == job['processStartTime']:
        try:
            if os.getpgid(pid) != pid:
                raise RuntimeError('Training process no longer owns its recorded process group')
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + min(max(timeout, 0), 15)
            while _process_start(pid) == job['processStartTime'] and time.monotonic() < deadline:
                time.sleep(.1)
            if _process_start(pid) == job['processStartTime']:
                os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    latest = json.loads(path.read_text())
    latest.update(status='cancelled', cancelRequested=True, finishedAt=time.time())
    atomic_json(path, latest)
    return latest


def install_checkpoint(model, checkpoint, base, destination, stage):
    expected = {'forge-wm': 'stage3-student', 'sana-wm': 'self-forcing-t121'}
    if expected.get(model) != stage:
        raise ValueError('Only the final matching distilled student can be activated by this adapter')
    checkpoint, base, destination = checkpoint.resolve(), base.resolve(), destination.resolve()
    if checkpoint.suffix != '.pt' or not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError('Expected a nonempty upstream .pt generator checkpoint')
    # weights_only prevents arbitrary pickle code in custom artifacts. Inspect
    # CPU tensors only; this command does not allocate a GPU.
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a tensor state dictionary')
    if model == 'forge-wm':
        state = state.get('generator', state.get('generator_ema', state))
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError('Invalid Forge generator state dictionary')
    else:
        state = state.get('generator', state)
        state = state.get('state_dict', state)
        if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
            raise ValueError('Invalid SANA student state dictionary')
    m = manifest(model)
    for name in m['weightFiles']:
        if name not in ('stage3/model.pt', 'sana_dit/model.pt') and not (base / name).is_file():
            raise ValueError('Base bundle is incomplete: ' + name)
    destination.mkdir(parents=True, exist_ok=False)
    target = 'stage3/model.pt' if model == 'forge-wm' else 'sana_dit/model.pt'
    for item in base.iterdir():
        if item.name != target.split('/')[0]:
            (destination / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (destination / target).parent.mkdir()
    shutil.copy2(checkpoint, destination / target)
    with checkpoint.open('rb') as checkpoint_file:
        digest = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
    record = {'model': model, 'stage': stage, 'sourceCommit': m['commit'], 'checkpointSha256': digest, 'gpuValidated': False, 'baseBundle': str(base)}
    atomic_json(destination / 'worlds-customization.json', record)
    return record


def select_bundle(model, bundle, output):
    bundle = bundle.resolve()
    record = json.loads((bundle / 'worlds-customization.json').read_text())
    if record['model'] != model or record['sourceCommit'] != manifest(model)['commit']:
        raise ValueError('Checkpoint bundle targets a different model revision')
    # A local environment fragment selects the next worker process. Does not
    # mutate a running model, start a service, or deploy anything.
    import shlex
    output.write_text(manifest(model)['prefix'] + '_WEIGHTS=' + shlex.quote(str(bundle)) + '\n')


def disable_bundle(model, output, base=None):
    import shlex
    prefix = manifest(model)['prefix'] + '_WEIGHTS'
    if base is None and output.is_file():
        tokens = shlex.split(output.read_text())
        if len(tokens) == 1 and tokens[0].startswith(prefix + '='):
            selected = Path(tokens[0].split('=', 1)[1])
            metadata = selected / 'worlds-customization.json'
            if metadata.is_file():
                record = json.loads(metadata.read_text())
                if record.get('model') == model:
                    base = record.get('baseBundle')
    if base is not None:
        base = Path(base).resolve()
        if not base.is_dir():
            raise ValueError('Original base checkpoint bundle is no longer available')
        output.write_text(prefix + '=' + shlex.quote(str(base)) + '\n')
    else:
        output.write_text('unset ' + prefix + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--model', choices=RECIPES, required=True)
    prepare.add_argument('--stage', required=True)
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--config', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--python', default=sys.executable)
    prepare.add_argument('--devices', type=int, default=8)
    run = sub.add_parser('run')
    run.add_argument('directory', type=Path)
    run.add_argument('--run', action='store_true')
    cancel = sub.add_parser('cancel')
    cancel.add_argument('directory', type=Path)
    install = sub.add_parser('install')
    install.add_argument('--model', choices=RECIPES, required=True)
    install.add_argument('--stage', required=True)
    install.add_argument('--checkpoint', type=Path, required=True)
    install.add_argument('--base', type=Path, required=True)
    install.add_argument('--destination', type=Path, required=True)
    for name in ('enable', 'disable'):
        action = sub.add_parser(name)
        action.add_argument('--model', choices=RECIPES, required=True)
        action.add_argument('--env-file', type=Path, required=True)
        if name == 'enable':
            action.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = create_job(args.model, args.stage, args.source, args.config, args.output, args.python, args.devices)
    elif args.command == 'run':
        result = run_job(args.directory, args.run)
    elif args.command == 'cancel':
        result = cancel_job(args.directory)
    elif args.command == 'install':
        result = install_checkpoint(args.model, args.checkpoint, args.base, args.destination, args.stage)
    elif args.command == 'enable':
        select_bundle(args.model, args.bundle, args.env_file)
        result = {'status': 'selected-for-next-worker-start', 'model': args.model}
    else:
        disable_bundle(args.model, args.env_file)
        result = {'status': 'base-checkpoint-selected-for-next-worker-start', 'model': args.model}
    print(json.dumps(result, indent=2))
    if result.get('status') == 'failed':
        raise SystemExit(result.get('exitCode') or 1)


if __name__ == '__main__':
    main()
