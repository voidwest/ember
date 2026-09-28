#!/usr/bin/env python3
"""Gate D coverage for the `scale` operation at every six-site site/phase/mode.

Consumes a verified six-site capture matrix made with the identical executable
(see validate_six_site_captures.py). For each site/phase it runs a scale(0.5)
intervention and a scale-then-restore intervention. The target capture must be
the pre-intervention row; a scaled run at a site with captured downstream sites
must be observable; the scale-then-restore run must reproduce every baseline
capture and output byte exactly. Requires a new output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from validate_six_site_captures import SITES, digest, read_captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    identities = json.loads((baseline / 'identities.json').read_text())
    assert json.loads((baseline / 'summary.json').read_text())['passed']
    for identity in identities.values():
        assert digest(Path(identity['path'])) == identity['sha256'], 'baseline identity changed'
    binary = identities['binary']['path']
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    (work / 'identities.json').write_text(json.dumps(identities, indent=2) + '\n')
    results = []

    def command(label, action, *parameters):
        argv = [binary, 'experiment', action, *map(str, parameters), '--json']
        (work / f'{label}-{action}.command.json').write_text(json.dumps(argv) + '\n')
        with (work / f'{label}-{action}.json').open('w') as out, (work / f'{label}-{action}.log').open('w') as err:
            subprocess.run(argv, stdout=out, stderr=err, check=True)
        return json.loads((work / f'{label}-{action}.json').read_text())

    def block(site, selector, index, operation):
        if operation == 'scale':
            body = 'operation = { kind = "scale", factor = 0.5 }'
        else:
            body = f'operation = {{ kind = "{operation}" }}'
        text = f'\n[[interventions]]\nid = "op-{index}"\nsite = "{site}"\n{body}\ntokens = {{ {selector} }}\n'
        if site in SITES[:4]:
            text += 'layers = [0]\n'  # per-layer sites (first four) carry layers
        return text

    for mode in ('reference', 'planned', 'planned-fused'):
        original = baseline / mode
        assert command(f'{mode}-baseline', 'verify', original)['ok']
        captures = read_captures(original)
        outputs = (original / 'outputs.jsonl').read_bytes()
        for site in SITES:
            for phase, selector in [('prefill', 'kind = "prompt-final"'),
                                    ('decode', 'kind = "generated-step", step = 1')]:
                capture_id = f'{site}-{phase}-selected-rows'
                expected, expected_data = captures[capture_id]
                for leg, operations in [('scaled', ('scale',)),
                                        ('scale-restored', ('scale', 'restore-original'))]:
                    label = f'{mode}-{site}-{phase}-{leg}'
                    spec = (baseline / f'{mode}.toml').read_text()
                    for index, operation in enumerate(operations):
                        spec += block(site, selector, index, operation)
                    spec_path = work / f'{label}.toml'
                    spec_path.write_text(spec)
                    bundle = work / label
                    command(label, 'run', spec_path, '--output', bundle)
                    assert command(label, 'verify', bundle)['ok']
                    observed = read_captures(bundle)
                    assert observed.keys() == captures.keys(), label
                    # The target capture is the original row before the scale runs.
                    assert observed[capture_id][1] == expected_data, label
                    events = [json.loads(line) for line in
                              (bundle / 'interventions/events.jsonl').read_text().splitlines()]
                    assert len(events) == len(operations), (label, events)
                    for index, event in enumerate(events):
                        assert event['intervention_id'] == f'op-{index}', label
                        assert event['operation'] == operations[index], (label, event)
                        assert event['site'] == site and event['positions'] == expected['positions'], label
                        assert event['applied'], label
                        assert event['snapshot_checksum'] == hashlib.sha256(expected_data).hexdigest(), label
                    equal_captures = all(observed[key][1] == value[1] for key, value in captures.items())
                    equal_outputs = (bundle / 'outputs.jsonl').read_bytes() == outputs
                    if leg == 'scale-restored':
                        assert equal_captures and equal_outputs, f'{label}: scale+restore was not exact'
                    elif site != 'logits':
                        # Sites before the last captured site must affect a downstream capture.
                        assert not equal_captures, f'{label}: scale had no observable effect'
                    results.append({'mode': mode, 'site': site, 'phase': phase, 'leg': leg,
                                    'capture_before_intervention_exact': True,
                                    'event_order_exact': True, 'captures_exact': equal_captures,
                                    'outputs_exact': equal_outputs})
                    (work / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
                print(f'{mode} {site} {phase}: scale observable; scale+restore exact', flush=True)
    (work / 'summary.json').write_text(json.dumps({'passed': True, 'cases': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
