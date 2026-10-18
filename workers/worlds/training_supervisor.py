"""Keep local distributed training bounded even when its control service dies.

Launched in a fresh process group by customization.run_job. The supervisor and
all torchrun ranks share that group; TERM then KILL covers descendants even if
the launcher exits first. No shell or network/provisioning API is used.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_start(pid):
    try:
        return Path('/proc/' + str(int(pid)) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def live_group_members(group):
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def clean_remaining_ranks(grace):
    group = os.getpgrp()
    if not live_group_members(group):
        return
    os.killpg(group, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while live_group_members(group) and time.monotonic() < deadline:
        time.sleep(.05)
    if live_group_members(group):
        # Include this group leader: do not report a successful job completion
        # while TERM-resistant descendants could still occupy GPUs.
        os.killpg(group, signal.SIGKILL)


def supervise(argv, parent_pid, parent_start, max_seconds, grace=10):
    if not argv or os.getpgrp() != os.getpid():
        raise ValueError('Training supervisor must own a fresh process group')
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = subprocess.Popen(argv)
    deadline = time.monotonic() + max_seconds
    try:
        while child.poll() is None:
            if stopping or time.monotonic() >= deadline or process_start(parent_pid) != parent_start:
                os.killpg(os.getpgrp(), signal.SIGTERM)
                try:
                    child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
                # Descendants can outlive torchrun. Keep our own group leader
                # present and send KILL to the complete group after grace.
                os.killpg(os.getpgrp(), signal.SIGKILL)
                return 137  # unreachable when the OS delivered KILL
            time.sleep(.1)
        clean_remaining_ranks(grace)
        return child.returncode
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--parent-start', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('--grace-seconds', type=float, default=10)
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not 0 < args.max_seconds <= 172800 or not 0 <= args.grace_seconds <= 15:
        parser.error('Invalid supervision timeout')
    raise SystemExit(supervise(argv, args.parent_pid, args.parent_start, args.max_seconds, args.grace_seconds))
