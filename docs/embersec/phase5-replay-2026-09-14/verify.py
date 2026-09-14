#!/usr/bin/env python3
"""Verify this saved replay package without model files or Rust."""
from pathlib import Path
import hashlib,json,struct
P=Path(__file__).resolve().parent
for line in (P/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
rows=[json.loads(s) for s in (P/'sweep_replay_1.jsonl').read_text().splitlines()]
assert len(rows)==576
assert (P/'sweep_replay_1.jsonl').read_bytes()==(P/'sweep_replay_2.jsonl').read_bytes()==(P/'validator_sweep.jsonl').read_bytes()
assert hashlib.sha256((P/'sweep_replay_1.jsonl').read_bytes()).hexdigest()=='fbf34d0d1f02f819d1141012c01ae7f90e2266ab9e4b336218960a5faf19abc6'
verdicts=json.loads((P/'validator_rows.json').read_text())
assert len(verdicts)==576
for r,v in zip(rows,verdicts):
 b=int(r['d_bits'],16)
 assert (b^(1<<r['bit']))==int(r['faulted_bits'],16)
 request=struct.unpack('<f',struct.pack('<f',r['d_requested']))[0]
 assert struct.unpack('<H',struct.pack('<e',request))[0]==b
 assert (v['dtype'],v['d_bits'],v['bit'])==(r['dtype'],b,r['bit'])
 assert v['accepted']==r['finite']
real=[r for r in rows if abs(r['d_requested'])!=1]
assert len(real)==512 and all(r['finite'] for r in real)
assert sum(not r['finite'] for r in rows)==4
inv=json.loads((P/'model_inventory.json').read_text());d=json.loads((P/'d_distribution.json').read_text());ind=json.loads((P/'independent_scan.json').read_text())
assert len(inv)==len(d)==len(ind)==7
assert sum(s['n_blocks'] for file in d.values() for s in file.values())==136307712
for name,v in inv.items():
 assert v['sha256']==v['sha256_after'] and v['stat_before']==v['stat_after']
 for dtype,s in d[name].items():
  i=ind[name][dtype];assert i['d_words']==s['n_blocks'] and i['tensors']==s['n_tensors']
  assert i['single_bit_nonfinite_eligible_words']==0
  assert all(round(s['frac_by_exponent_bucket'][str(e)]*s['n_blocks'])==i['exponent_counts'][e] for e in range(32))
k=json.loads((P/'kernel_result.json').read_text());assert sum(v['samples'] for v in k['loader_checks'].values())==45
assert all('0 mismatched' in v['output'] for v in k['loader_checks'].values())
print('PASS: package hashes, exact repeated sweep bytes, 576 independent bit checks, direct validator verdicts, seven file identities, independent exponent counts, and 45 loader spot checks.')
