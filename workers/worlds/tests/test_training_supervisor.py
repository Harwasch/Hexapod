"""Real CPU subprocess tests for training deadline and service-death cleanup."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_supervisor import process_start

SCRIPT = Path(__file__).resolve().parents[1] / 'training_supervisor.py'


class TrainingSupervisorTest(unittest.TestCase):
    def command(self, timeout, parent_start=None, child=None):
        return [sys.executable, str(SCRIPT), '--parent-pid', str(os.getpid()),
            '--parent-start', parent_start or process_start(os.getpid()),
            '--max-seconds', str(timeout), '--grace-seconds', '.1', '--',
            sys.executable, '-c', child or 'import time; time.sleep(30)']

    def test_successful_child_preserves_exit_code(self):
        process = subprocess.Popen(self.command(5, child='raise SystemExit(0)'), start_new_session=True)
        self.assertEqual(process.wait(timeout=5), 0)

    def test_deadline_kills_training_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            pidfile = Path(temporary) / 'pid'
            child = 'import os,signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid())); time.sleep(30)'
            process = subprocess.Popen(self.command(.3, child=child), start_new_session=True)
            try:
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
                child_pid = int(pidfile.read_text())
                stat = Path('/proc') / str(child_pid) / 'stat'
                if stat.exists():
                    self.assertEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'Z')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()

    def test_launcher_exit_cannot_leave_descendant_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            pidfile = Path(temporary) / 'grandchild'
            descendant = 'import time; time.sleep(30)'
            child = 'import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,"-c",' + repr(descendant) + ']); pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(p.pid))'
            process = subprocess.Popen(self.command(30, child=child), start_new_session=True)
            try:
                self.assertEqual(process.wait(timeout=5), 0)
                stat = Path('/proc') / pidfile.read_text() / 'stat'
                if stat.exists():
                    self.assertEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'Z')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()

    def test_changed_parent_identity_stops_child_before_deadline(self):
        process = subprocess.Popen(self.command(30, parent_start='not-current-start'), start_new_session=True)
        try:
            self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL); process.wait()


if __name__ == '__main__': unittest.main()
