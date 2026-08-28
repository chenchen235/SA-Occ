#!/usr/bin/env python3
"""Extract multiple named nuScenes scenes into one timestamp-sorted infos pkl."""
import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('scene_names', nargs='+')
    parser.add_argument('--input', action='append', required=True, type=Path)
    args = parser.parse_args()

    requested = set(args.scene_names)
    subset, metadata = [], None
    for source in args.input:
        with source.open('rb') as handle:
            data = pickle.load(handle)
        infos = data['infos'] if isinstance(data, dict) else data
        subset.extend(info for info in infos if info.get('scene_name') in requested)
        if metadata is None and isinstance(data, dict):
            metadata = data.get('metadata')
    found = {info['scene_name'] for info in subset}
    missing = requested - found
    if missing:
        raise SystemExit(f'Missing scenes: {sorted(missing)}')
    subset.sort(key=lambda info: info['timestamp'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        pickle.dump({'infos': subset, 'metadata': metadata or {}}, handle)
    counts = {scene: sum(info['scene_name'] == scene for info in subset) for scene in sorted(requested)}
    print(f'wrote {len(subset)} frames to {args.output}: {counts}')


if __name__ == '__main__':
    main()
