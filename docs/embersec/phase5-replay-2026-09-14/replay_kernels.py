import datetime,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
SOURCE=Path('/tmp/embersec-phase5-replay-source')
CRATE=SOURCE/'docs/embersec/phase5-correction-2026-09-03/sweep'
ORIGINAL=SOURCE/'docs/embersec/phase5-correction-2026-09-03/sweep.jsonl'
env={k:v for k,v in os.environ.items() if not k.startswith('EMBER_')};env.update(EMBER_VERIFY_QUANT='0',RAYON_NUM_THREADS='4',CARGO_BUILD_JOBS='4')
commands=[]
def run(args,label,seconds=900):
 start=time.monotonic();p=subprocess.run(args,cwd=SOURCE,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=seconds)
 (OUT/(label+'.stdout')).write_text(p.stdout);(OUT/(label+'.stderr')).write_text(p.stderr)
 commands.append({'label':label,'argv':[str(x) for x in args],'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start})
 (OUT/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label+' failed; see saved stderr')
 return p
build=['cargo','build','--offline','--locked','--manifest-path',str(CRATE/'Cargo.toml')]
run(build+['--bin','sweep'],'locked_build')
shutil.copy2(CRATE/'Cargo.lock',OUT/'Cargo.lock');shutil.copy2(CRATE/'Cargo.toml',OUT/'sweep.Cargo.toml')
binary=CRATE/'target/debug/sweep'
for i in [1,2]:run([str(binary),str(OUT/f'sweep_replay_{i}.jsonl')],f'sweep_run_{i}',60)
old=ORIGINAL.read_bytes();r1=(OUT/'sweep_replay_1.jsonl').read_bytes();r2=(OUT/'sweep_replay_2.jsonl').read_bytes()
summary={'original_sha256':hashlib.sha256(old).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256(r1).hexdigest(),'exact_bytes_original':r1==old,'exact_bytes_repeat':r1==r2,'rows':len(r1.splitlines())}
if r1!=old:
 a=[json.loads(x) for x in old.splitlines()];b=[json.loads(x) for x in r1.splitlines()];summary['differences']=[{'row':i,'fields':{k:[x[k],y.get(k)] for k in x if x[k]!=y.get(k)}} for i,(x,y) in enumerate(zip(a,b)) if x!=y]
(OUT/'kernel_result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
shutil.copy2(OUT/'validate_faults.rs',CRATE/'src/bin/validate_faults.rs')
run(build+['--bin','validate_faults','--bin','validate_scales'],'validation_build')
v=run([str(CRATE/'target/debug/validate_faults'),str(OUT/'validator_sweep.jsonl')],'validator_run',60)
vr=[]
for line in v.stderr.splitlines():
 d,b,bit,accepted=line.split();vr.append({'dtype':d,'d_bits':int(b),'bit':int(bit),'accepted':accepted=='true'})
rows=[json.loads(x) for x in r1.splitlines()]
assert len(vr)==len(rows)==576
for x,y in zip(vr,rows):assert (x['dtype'],x['d_bits'],x['bit'])==(y['dtype'],int(y['d_bits'],16),y['bit'])
summary['validator_outputs_exact_replay']=(OUT/'validator_sweep.jsonl').read_bytes()==r1
summary['validator_real_accepted']=sum(x['accepted'] for x,y in zip(vr,rows) if abs(y['d_requested'])!=1)
summary['validator_nonfinite_rejected']=sum(not x['accepted'] for x,y in zip(vr,rows) if not y['finite'])
summary['validator_mismatch_count']=sum(x['accepted']!=y['finite'] for x,y in zip(vr,rows))
(OUT/'validator_rows.json').write_text(json.dumps(vr,indent=2)+'\n')
samples=json.loads((OUT/'d_samples.json').read_text());loads={}
for i,(name,values) in enumerate(samples.items()):
 sample_path=OUT/f'loader_samples_{i}.json';sample_path.write_text(json.dumps(values,indent=2)+'\n')
 p=run([str(CRATE/'target/debug/validate_scales'),str(Path('/home/west/ember')/name),str(sample_path)],f'loader_check_{i}',180)
 loads[name]={'samples':len(values),'output':p.stdout.strip()}
summary['loader_checks']=loads;summary['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'kernel_result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
