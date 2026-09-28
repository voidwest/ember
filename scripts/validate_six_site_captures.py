#!/usr/bin/env python3
"""Real-model v1 capture matrix: six sites, prefill/decode, selected/full rows.

Requires an explicit model and tokenizer; missing inputs never count as a pass.
Uses only the Python standard library to compare serialized F32 rows bitwise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

SITES = ('residual-pre-attention', 'attention-output', 'mlp-output',
         'residual-post-mlp', 'final-norm-output', 'logits')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_captures(bundle):
    raw = (bundle / 'captures/tensors.safetensors').read_bytes()
    header_size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    data = raw[8 + header_size:]
    result = {}
    for line in (bundle / 'captures/index.jsonl').read_text().splitlines():
        entry = json.loads(line)
        tensor = header[entry['tensor_name']]
        assert tensor['dtype'] == entry['dtype'] == 'F32'
        assert tensor['shape'] == entry['shape']
        begin, end = tensor['data_offsets']
        result[entry['capture_id']] = (entry, data[begin:end])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--ember', type=Path, default=Path('target/release/ember'))
    args = parser.parse_args()
    model, tokenizer, binary = (p.resolve(strict=True) for p in
                                 (args.model, args.tokenizer, args.ember))
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    identities = {name: {'path': str(path), 'sha256': digest(path)}
                  for name, path in [('model', model), ('tokenizer', tokenizer), ('binary', binary)]}
    (work / 'identities.json').write_text(json.dumps(identities, indent=2) + '\n')
    comparisons = []
    for mode in ('reference', 'planned', 'planned-fused'):
        bundle = work / mode
        spec = f'''schema = "ember.experiment.v1"
[experiment]
name = "six-site-captures"
seed = 42
[model]
path = {json.dumps(str(model))}
expected_sha256 = "{identities['model']['sha256']}"
tokenizer = {json.dumps(str(tokenizer))}
tokenizer_expected_sha256 = "{identities['tokenizer']['sha256']}"
[execution]
mode = "{mode}"
threads = 4
deterministic = true
[generation]
max_new_tokens = 3
temperature = 0.0
[[inputs]]
id = "mixed"
text = "The Arabic word كتاب means"
[output]
directory = {json.dumps(str(bundle))}
'''
        for site in SITES:
            for phase, selector in [('prefill', 'kind = "prompt-final"'),
                                    ('decode', 'kind = "generated-step", step = 1')]:
                for storage in ('selected-rows', 'full-tensor'):
                    spec += f'''\n[[captures]]
id = "{site}-{phase}-{storage}"
site = "{site}"
'''
                    if site in SITES[:4]:
                        spec += 'layers = [0]\n'
                    spec += f'storage = "{storage}"\ntokens = {{ {selector} }}\n'
        spec_path = work / f'{mode}.toml'
        spec_path.write_text(spec)
        for action, parameters in [('run', [spec_path]), ('verify', [bundle, '--model', model, '--tokenizer', tokenizer])]:
            argv = [str(binary), 'experiment', action, *map(str, parameters), '--json']
            (work / f'{mode}-{action}.command.json').write_text(json.dumps(argv) + '\n')
            with (work / f'{mode}-{action}.json').open('w') as out, (work / f'{mode}-{action}.log').open('w') as err:
                subprocess.run(argv, stdout=out, stderr=err, check=True)
            if action == 'verify':
                assert json.loads((work / f'{mode}-{action}.json').read_text())['ok']
        captures = read_captures(bundle)
        assert len(captures) == 24, f'{mode}: missing captures'
        for site in SITES:
            for phase in ('prefill', 'decode'):
                selected, selected_data = captures[f'{site}-{phase}-selected-rows']
                full, full_data = captures[f'{site}-{phase}-full-tensor']
                assert selected['shape'][0] == 1
                assert selected['shape'][1] == full['shape'][1]
                stride = selected['shape'][1] * 4
                index = full['positions'].index(selected['positions'][0])
                assert selected_data == full_data[index * stride:(index + 1) * stride], (mode, site, phase)
                assert all(abs(v[0]) < float('inf') for v in struct.iter_unpack('<f', full_data))
                if phase == 'decode' or site in SITES[4:]:
                    assert full['shape'][0] == 1
                else:
                    assert full['shape'][0] > 1
                if phase == 'decode':
                    prompt, _ = captures[f'{site}-prefill-selected-rows']
                    assert selected['positions'] == [prompt['positions'][0] + 1]
                comparisons.append({'mode': mode, 'site': site, 'phase': phase,
                                    'selected_position': selected['positions'][0],
                                    'full_shape': full['shape'], 'exact': True})
        print(f'{mode}: all 12 selected/full row comparisons exact; deep verification passed', flush=True)
    (work / 'summary.json').write_text(json.dumps({'passed': True, 'comparisons': comparisons}, indent=2) + '\n')


if __name__ == '__main__':
    main()
