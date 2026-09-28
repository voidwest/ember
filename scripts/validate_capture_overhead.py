#!/usr/bin/env python3
"""Measure selected-row vs full-tensor capture overhead on a real model.

Runs the same v1 experiment three ways on a pinned model + tokenizer:
no captures, selected-row captures, and full-tensor captures at all six
hook sites (prefill prompt-final and decode generated-step 1). For each
variant it reports the median wall-clock time, decode throughput, peak RSS,
and the bundle payload size. Requires a new output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

SITES = ('residual-pre-attention', 'attention-output', 'mlp-output',
         'residual-post-mlp', 'final-norm-output', 'logits')
SELECTORS = (('prefill', 'kind = "prompt-final"'),
             ('decode', 'kind = "generated-step", step = 1'))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def spec_text(model, model_sha, tokenizer, tokenizer_sha, work, tokens, storage):
    text = f'''schema = "ember.experiment.v1"
[experiment]
name = "capture-overhead"
seed = 42
[model]
path = {json.dumps(str(model))}
expected_sha256 = "{model_sha}"
tokenizer = {json.dumps(str(tokenizer))}
tokenizer_expected_sha256 = "{tokenizer_sha}"
[execution]
mode = "reference"
threads = 4
deterministic = true
[generation]
max_new_tokens = {tokens}
temperature = 0.0
[[inputs]]
id = "mixed"
text = "The Arabic word كتاب means"
[output]
directory = {json.dumps(str(work))}
'''
    if storage is None:
        return text
    for site in SITES:
        for phase, selector in SELECTORS:
            text += f'\n[[captures]]\nid = "{site}-{phase}"\nsite = "{site}"\n'
            if site in SITES[:4]:
                text += 'layers = [0]\n'
            text += f'storage = "{storage}"\ntokens = {{ {selector} }}\n'
    return text


def run_once(binary, spec_path, bundle):
    argv = [str(binary), 'experiment', 'run', str(spec_path), '--output', str(bundle), '--json']
    with open(os.devnull, 'wb') as devnull:
        process = subprocess.Popen(argv, stdout=devnull, stderr=devnull)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    if process.returncode:
        raise RuntimeError(f'experiment run failed for {spec_path}')
    runtime = json.loads((bundle / 'runtime.json').read_text())
    payload = sum(f.stat().st_size for f in bundle.rglob('*') if f.is_file())
    rss = int(usage.ru_maxrss * (1024 if sys.platform == 'linux' else 1))
    return {'wall_clock_ms': runtime['wall_clock_ms'],
            'decode_tps': runtime['decode_throughput_tps'],
            'rss_bytes': rss, 'bundle_bytes': payload}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--ember', type=Path, default=Path('target/release/ember'))
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    model, tokenizer, binary = (p.resolve(strict=True) for p in
                                (args.model, args.tokenizer, args.ember))
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    identities = {name: {'path': str(path), 'sha256': digest(path)}
                  for name, path in [('model', model), ('tokenizer', tokenizer), ('binary', binary)]}
    (work / 'identities.json').write_text(json.dumps(identities, indent=2) + '\n')
    report = {'scope': 'selected-row vs full-tensor capture overhead; wall clock, throughput, peak RSS',
              'tokens': args.tokens, 'repeats': args.repeats, 'identities': identities, 'variants': {}}
    for label, storage in [('none', None), ('selected-rows', 'selected-rows'), ('full-tensor', 'full-tensor')]:
        samples = []
        for index in range(args.repeats):
            bundle = work / f'{label}-{index}'
            spec_path = work / f'{label}-{index}.toml'
            spec_path.write_text(spec_text(model, identities['model']['sha256'], tokenizer,
                                           identities['tokenizer']['sha256'], bundle, args.tokens, storage))
            samples.append(run_once(binary, spec_path, bundle))
            (work / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
        median = {key: statistics.median(s[key] for s in samples) for key in samples[0]}
        report['variants'][label] = {'median': median, 'samples': samples}
        print(f'{label:14} wall {median["wall_clock_ms"]:8.2f} ms  tps {median["decode_tps"]:7.2f}  '
              f'rss {median["rss_bytes"]}  bundle {median["bundle_bytes"]}', flush=True)
    base = report['variants']['none']['median']
    for label in ('selected-rows', 'full-tensor'):
        median = report['variants'][label]['median']
        report['variants'][label]['wall_clock_overhead_percent'] = \
            100.0 * (median['wall_clock_ms'] / base['wall_clock_ms'] - 1)
        report['variants'][label]['throughput_change_percent'] = \
            100.0 * (median['decode_tps'] / base['decode_tps'] - 1)
        report['variants'][label]['bundle_size_change_bytes'] = median['bundle_bytes'] - base['bundle_bytes']
    (work / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
