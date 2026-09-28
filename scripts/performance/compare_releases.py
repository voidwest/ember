#!/usr/bin/env python3
"""Compare explicit CPU release binaries on the same pinned local models.

Runs the historical K-quant matrix: 64 evaluations, one warmup, three measured
repetitions; reference/planned/planned-fused; 1/2/4/8 threads. Each cell uses
baseline/candidate/candidate/baseline process order. POSIX wait4 records each
process's own peak RSS. Requires a new output directory; no stale-run skipping.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def identity(path):
    path = path.resolve(strict=True)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-cwd', type=Path, required=True)
    parser.add_argument('--candidate-cwd', type=Path, required=True)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform not in ('darwin', 'linux'):
        parser.error('RSS units are defined only for macOS and Linux')
    binaries = {'baseline': identity(args.baseline), 'candidate': identity(args.candidate)}
    models = [identity(path) for path in args.model]
    if len({Path(m['path']).name for m in models}) != len(models):
        parser.error('model basenames must be unique')
    cwd = {'baseline': str(args.baseline_cwd.resolve(strict=True)),
           'candidate': str(args.candidate_cwd.resolve(strict=True))}
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    # Freeze the protocol and all identities before observing any timing.
    protocol = {'binaries': binaries, 'models': models, 'cwd': cwd,
                'platform': platform.platform(), 'tokens': 64, 'warmups': 1,
                'repetitions': 3, 'threads': [1, 2, 4, 8],
                'modes': ['reference', 'planned', 'planned-fused'],
                'order': ['baseline', 'candidate', 'candidate', 'baseline'],
                'throughput_regression_limit_percent': 3,
                'rss_regression_limit_percent': 3,
                'scope': 'K-quant model-only decode; not complete release Gate H',
                'rss': 'wait4 child ru_maxrss, normalized to bytes'}
    (work / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    cells = []
    for model in models:
        model_name = Path(model['path']).stem
        for mode in protocol['modes']:
            for threads in protocol['threads']:
                samples = []
                label = f'{model_name}-{mode}-t{threads}'
                for index, name in enumerate(protocol['order']):
                    # Refuse replacement of a binary while the matrix is running.
                    assert identity(Path(binaries[name]['path'])) == binaries[name], name
                    prefix = work / f'{label}-{index}-{name}'
                    argv = [binaries[name]['path'], 'bench-decode', '--model', model['path'],
                            '--arch', 'llama', '--execution', mode, '--tokens', '64',
                            '--warmups', '1', '--repetitions', '3', '--token-id', '1']
                    env = dict(os.environ)
                    env['RAYON_NUM_THREADS'] = str(threads)
                    Path(str(prefix) + '.command.json').write_text(json.dumps(
                        {'argv': argv, 'cwd': cwd[name], 'RAYON_NUM_THREADS': str(threads)}, indent=2) + '\n')
                    with Path(str(prefix) + '.json').open('w') as out, Path(str(prefix) + '.log').open('w') as err:
                        process = subprocess.Popen(argv, cwd=cwd[name], env=env, stdout=out, stderr=err)
                        _, status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(status)
                    rss = int(usage.ru_maxrss * (1024 if sys.platform == 'linux' else 1))
                    Path(str(prefix) + '.resource.json').write_text(json.dumps(
                        {'returncode': process.returncode, 'rss_bytes': rss,
                         'user_seconds': usage.ru_utime, 'system_seconds': usage.ru_stime}, indent=2) + '\n')
                    if process.returncode:
                        raise RuntimeError(f'{prefix}: process exited {process.returncode}; inspect log')
                    data = json.loads(Path(str(prefix) + '.json').read_text())
                    assert data['threads'] == threads
                    assert data['tokens'] == 64 and data['warmups'] == 1 and data['repetitions'] == 3
                    tps = data['median_tokens_per_second']
                    assert math.isfinite(tps) and tps > 0 and rss > 0
                    assert data['k_compressed_bytes'] > 0
                    assert data['k_expanded_bytes'] == data['k_fallback_count'] == 0
                    samples.append({'binary': name, 'median_tps': tps, 'rss_bytes': rss})
                    print(f'{label} {index + 1}/4 {name}: {tps:.3f} tokens/s; RSS {rss}', flush=True)
                result = {'model': model, 'mode': mode, 'threads': threads, 'samples': samples}
                for name in binaries:
                    subset = [sample for sample in samples if sample['binary'] == name]
                    result[name] = {'median_tps': statistics.median(sample['median_tps'] for sample in subset),
                                    'max_rss_bytes': max(sample['rss_bytes'] for sample in subset)}
                result['throughput_change_percent'] = 100 * (result['candidate']['median_tps'] / result['baseline']['median_tps'] - 1)
                result['rss_change_percent'] = 100 * (result['candidate']['max_rss_bytes'] / result['baseline']['max_rss_bytes'] - 1)
                result['within_limits'] = result['throughput_change_percent'] >= -3 and result['rss_change_percent'] <= 3
                cells.append(result)
                (work / 'progress.json').write_text(json.dumps(cells, indent=2) + '\n')
    (work / 'summary.json').write_text(json.dumps(
        {'complete': True, 'within_limits': all(cell['within_limits'] for cell in cells), 'cells': cells}, indent=2) + '\n')


if __name__ == '__main__':
    main()
