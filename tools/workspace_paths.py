#!/usr/bin/env python3
"""Read configs/paths.json and make its paths valid on this machine.

The config is written with the Linux mount point of the data drive. The same
drive gets a letter under Windows, so every path in the file is one prefix
substitution away from correct. Translating here keeps a single config that
works on both, instead of a second file to keep in sync. Set DGM_LINUX_MOUNT
if the drive is mounted somewhere else.
"""
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_MOUNT = os.environ.get('DGM_LINUX_MOUNT', '/media/mt-pc-0099/One Touch')


def _windows_root():
    """Which drive holds the data on this machine, if any."""
    for drive in ('E:', 'D:', 'F:'):
        if os.path.isdir(os.path.join(drive + os.sep, 'nuscenes')):
            return drive
    return None


def translate(value, root=None):
    if isinstance(value, dict):
        return {k: translate(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [translate(v, root) for v in value]
    if not isinstance(value, str) or not value.startswith(LINUX_MOUNT):
        return value
    if os.name != 'nt':
        return value
    root = root or _windows_root()
    if root is None:
        return value
    return (root + value[len(LINUX_MOUNT):]).replace('/', os.sep)


def load(path=None):
    path = path or os.path.join(ROOT, 'configs', 'paths.json')
    with io.open(path, encoding='utf-8') as f:
        return translate(json.load(f))


if __name__ == '__main__':
    for k, v in load().items():
        print('%-12s %s' % (k, v))
