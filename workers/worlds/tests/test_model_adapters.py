"""CPU wiring checks; these do not simulate or certify GPU inference."""
import ast
import importlib
import json
from pathlib import Path
import tempfile
import subprocess
import threading
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters import selected_adapter
from model_bootstrap import patch_matrix, PACKAGES
from matrix_game.engine import action_vectors
from forgewm.engine import ACTION_MAP as FORGE_ACTIONS
from sana_wm.engine import ACTION_MAP as SANA_ACTIONS
from model_support import ModelClient
from customization import create_job, run_job, select_bundle, cancel_job, _process_start, training_environment, disable_bundle


class ModelAdaptersTest(unittest.TestCase):
    def test_offline_registry_imports_without_torch(self):
        for name in ('forge-wm', 'matrix-game-3', 'sana-wm'):
            adapter = selected_adapter(name)
            self.assertEqual(adapter.model_id, name)
            self.assertFalse(adapter.ready()[0])
            self.assertTrue(adapter.capabilities['input']['requiredImage'])
            self.assertFalse(adapter.metadata()['gpuInferenceVerified'])
            adapter.close()
        with self.assertRaises(ValueError):
            selected_adapter('invented')

    def test_native_control_schemas_match_upstream(self):
        for action, index in [('forward', 0), ('backward', 1), ('left', 2), ('right', 3)]:
            keys, mouse = action_vectors(action)
            self.assertEqual(len(keys), 6)
            self.assertEqual(keys[index], 1.)
            self.assertEqual(sum(keys), 1.)
            self.assertEqual(mouse, [0., 0.])
        self.assertEqual(action_vectors('stop'), ([0.] * 6, [0.] * 2))
        self.assertEqual(action_vectors('look_left')[1], [0., -.1])
        self.assertEqual(FORGE_ACTIONS['backward'], 'back')
        # SANA uses j/l for strafe, a/d for yaw (opposite naive WASD map).
        self.assertEqual(SANA_ACTIONS['left'], 'j')
        self.assertEqual(SANA_ACTIONS['look_left'], 'a')

    def test_source_patch_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pipeline').mkdir()
            p = root / 'pipeline/inference_pipeline.py'
            p.write_text('def test():\n    if True:\n        if True:\n            exit()\n')
            patch_matrix(root)
            self.assertIn('return video', p.read_text())
            patch_matrix(root)
            ast.parse(p.read_text())
            p.write_text('exit()\n')
            with self.assertRaises(RuntimeError):
                patch_matrix(root)

    def test_output_fps_and_private_paths(self):
        client = ModelClient.__new__(ModelClient)
        client.expected_fps = 17
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = directory / 'frame.png'
            frame.write_bytes(b'frame')
            result = {'framePaths': [str(frame)], 'continuationPath': str(frame), 'fps': 17, 'nativeKVContinuity': False}
            client._validate_output(result, directory)
            with self.assertRaises(RuntimeError):
                client._validate_output(dict(result, fps=24), directory)
            with self.assertRaises(RuntimeError):
                client._validate_output(dict(result, continuationPath='/etc/passwd'), directory)

    def test_readiness_checks_all_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = selected_adapter('sana-wm')
            adapter.source, adapter.weights = root / 'source', root / 'weights'
            adapter.source.mkdir(); adapter.weights.mkdir()
            (adapter.source / '.worlds-source.json').write_text(json.dumps({'commit': adapter.manifest['commit'], 'patchVersion': 1}))
            for base, files in ((adapter.source, adapter.manifest['sourceFiles']), (adapter.weights, adapter.manifest['weightFiles'])):
                for filename in files:
                    path = base / filename; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({'weight_map': {'weight': 'part.safetensors'}}) if filename.endswith('.index.json') else 'fixture')
            self.assertFalse(adapter.ready()[0])
            for folder in ('gemma2_2b', 'gemma3_12b'):
                (adapter.weights / folder / 'part.safetensors').write_bytes(b'fixture')
            self.assertTrue(adapter.ready()[0])
            custom = root / 'custom'
            custom.mkdir()
            for item in adapter.weights.iterdir():
                (custom / item.name).symlink_to(item, target_is_directory=item.is_dir())
            adapter.weights = custom
            self.assertTrue(adapter.ready()[0], 'Installed custom bundles may link immutable base shards')
            adapter.close()

    def test_cancel_terminates_only_recorded_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
            record = {'status':'running','pid':process.pid,'processStartTime':_process_start(process.pid)}
            (directory / 'job.json').write_text(json.dumps(record))
            reaper = threading.Thread(target=process.wait)
            reaper.start()
            try:
                result = cancel_job(directory, timeout=.5)
                reaper.join(2)
                self.assertEqual(result['status'], 'cancelled')
                self.assertIsNotNone(process.poll())
                self.assertTrue((directory / '.cancel-requested').is_file())
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()

    def test_cancel_does_not_signal_recycled_pid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'job.json').write_text(json.dumps({'status':'running','pid':123,'processStartTime':'old'}))
            with patch('customization._process_start', return_value='new'), patch('customization.os.killpg') as kill:
                cancel_job(directory, timeout=0)
                kill.assert_not_called()

    def test_training_environment_does_not_forward_service_credentials(self):
        env = training_environment({'PATH':'/bin','HOME':'/home/worker','CUDA_VISIBLE_DEVICES':'0,1',
            'WORLD_CUSTOMIZATION_TOKEN':'secret','WORLD_GATEWAY_TOKEN':'secret','RUNPOD_API_KEY':'secret',
            'HF_TOKEN':'secret','AWS_SECRET_ACCESS_KEY':'secret','PYTHONPATH':'/untrusted'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertFalse(any('TOKEN' in key or 'KEY' in key for key in env))
        self.assertNotIn('PYTHONPATH', env)

    def test_sana_training_plan_resolves_local_encoder_without_leaking_wrapper_config(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'source'; source.mkdir()
            m = selected_adapter('sana-wm').manifest
            (source / '.worlds-source.json').write_text(json.dumps({'commit':m['commit'],'patchVersion':1}))
            script = source / 'train_video_scripts/train_longsana.py'
            script.parent.mkdir(); script.write_text('# fixture upstream entry')
            data = root / 'data'; data.mkdir()
            gemma = root / 'gemma'; gemma.mkdir()
            for filename in ('config.json','tokenizer.json'): (gemma / filename).write_text('{}')
            config = root / 'input.yaml'
            config.write_text(yaml.safe_dump({'worlds_components':{'gemma2':str(gemma)},'data_path':str(data)}))
            job = create_job('sana-wm','ode',source,config,root / 'job',sys.executable,8)
            self.assertIn('--gemma2',job['argv'])
            self.assertIn(str(gemma),job['argv'])
            self.assertTrue(any(value.endswith('/training_entry.py') for value in job['argv']))
            actual_config = yaml.safe_load((root / 'job/config.yaml').read_text())
            self.assertNotIn('worlds_components',actual_config)
            self.assertEqual(actual_config['work_dir'],str(root / 'job/checkpoints'))
            # Dry run does not even perform the CUDA prerequisite subprocess.
            with patch('customization.subprocess.run') as run:
                self.assertFalse(run_job(root / 'job')['executionRequested'])
                run.assert_not_called()

    def test_disable_restores_recorded_nondefault_base_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base = root / 'custom base location'; base.mkdir()
            selected = root / 'selected'; selected.mkdir()
            m = selected_adapter('forge-wm').manifest
            (selected / 'worlds-customization.json').write_text(json.dumps({'model':'forge-wm','sourceCommit':m['commit'],'baseBundle':str(base)}))
            env = root / 'selection.env'
            select_bundle('forge-wm',selected,env)
            disable_bundle('forge-wm',env)
            import shlex
            self.assertEqual(shlex.split(env.read_text()), ['FORGEWM_WEIGHTS=' + str(base)])

    def test_training_refuses_unsupported_model_before_execution(self):
        with self.assertRaises(ValueError):
            create_job('matrix-game-3', 'lora', Path('/tmp'), Path('/tmp/config'), Path('/tmp/output'), sys.executable, 8)

    def test_selection_checks_revision_and_only_writes_local_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = root / 'custom.env'
            m = selected_adapter('forge-wm').manifest
            (root / 'worlds-customization.json').write_text(json.dumps({'model':'forge-wm','sourceCommit':m['commit']}))
            select_bundle('forge-wm', root, env)
            self.assertIn('FORGEWM_WEIGHTS=', env.read_text())
            with self.assertRaises(ValueError):
                select_bundle('sana-wm', root, env)


if __name__ == '__main__':
    unittest.main()
