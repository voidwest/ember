#!/usr/bin/env python3
"""Reproduce the archived Phase V experiment using a captured dependency lock.

Run from a clone of voidwest/ember. Outputs go to a NEW directory. Model files
are read-only. The original 2026-09-03 artifact directory is never modified.
"""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
COMMIT='ae550f0fbcf9dea1a5507f5695b077d2cc77d02b'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--offline',action='store_true');args=p.parse_args()
repo=args.repo.resolve();models=args.models.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
source=out/'source';source.mkdir();archive=out/'source.tar'
with archive.open('wb') as f:subprocess.run(['git','-C',str(repo),'archive',COMMIT],stdout=f,check=True)
with tarfile.open(archive) as f:f.extractall(source,filter='data')
legacy=source/'docs/embersec/phase5-correction-2026-09-03';crate=legacy/'sweep'
manifest=crate/'Cargo.toml';manifest.write_text(manifest.read_text().replace('/home/west/ember',str(source)))
shutil.copyfile(HERE/'Cargo.lock',crate/'Cargo.lock')
shutil.copyfile(HERE/'validate_faults.rs',crate/'src/bin/validate_faults.rs')
env={k:v for k,v in os.environ.items() if not k.startswith('EMBER_')};env.update(EMBER_VERIFY_QUANT='0',RAYON_NUM_THREADS='4',CARGO_BUILD_JOBS='4')
commands=[]
def run(cmd,label):
 t=time.monotonic()
 with (out/(label+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 commands.append({'argv':cmd,'returncode':r.returncode,'seconds':time.monotonic()-t});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');r.check_returncode()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
metadata={'source_commit':COMMIT,'rustc':subprocess.check_output(['rustc','-Vv'],text=True),'cargo':subprocess.check_output(['cargo','-V'],text=True),'environment':{k:env[k] for k in ['EMBER_VERIFY_QUANT','RAYON_NUM_THREADS','CARGO_BUILD_JOBS']}}
(out/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
run(['cargo','build','--locked']+(['--offline'] if args.offline else [])+['--manifest-path',str(manifest),'--bin','sweep','--bin','validate_faults'],'build')
binary=crate/'target/debug/sweep'
for i in [1,2]:run([str(binary),str(out/f'sweep_{i}.jsonl')],f'sweep_{i}')
run([str(crate/'target/debug/validate_faults'),str(out/'validator_sweep.jsonl')],'validator')
reference=json.loads((legacy/'d_distribution.json').read_text());paths=[models/n for n in reference]
before={p.name:sha(p) for p in paths}
run(['python3',str(legacy/'gguf_d_scan.py'),*[str(p) for p in paths],'--out',str(out/'d_distribution.json'),'--samples',str(out/'d_samples.json')],'scan')
after={p.name:sha(p) for p in paths};(out/'model_hashes.json').write_text(json.dumps(after,indent=2)+'\n')
r={'sweep_exact_original':(out/'sweep_1.jsonl').read_bytes()==(legacy/'sweep.jsonl').read_bytes(),'sweep_exact_repeat':(out/'sweep_1.jsonl').read_bytes()==(out/'sweep_2.jsonl').read_bytes(),'validator_outputs_exact':(out/'validator_sweep.jsonl').read_bytes()==(out/'sweep_1.jsonl').read_bytes(),'distribution_exact':json.loads((out/'d_distribution.json').read_text())==reference,'model_hashes_stable':before==after,'binary_sha256':sha(binary),'lock_sha256':sha(crate/'Cargo.lock')}
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if not all(v for v in r.values() if isinstance(v,bool)):raise SystemExit(1)
