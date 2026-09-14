import datetime,hashlib,importlib.util,json,os,time
from pathlib import Path
ROOT=Path('/home/west/ember'); OUT=ROOT/'research/embersec/phase5_replay_20260914'
SRC=Path('/tmp/embersec-phase5-replay-source/docs/embersec/phase5-correction-2026-09-03')
spec=importlib.util.spec_from_file_location('scan',SRC/'gguf_d_scan.py');scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
old=json.loads((SRC/'d_distribution.json').read_text()); result={};records={};samples={}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
t0=time.monotonic()
for n in old:
 p=ROOT/n; before=identity(p); h=digest(p)
 data,samp=scan.scan_model(str(p)); h2=digest(p); after=identity(p)
 result[n]=data;samples[n]=samp; records[n]={'sha256':h,'sha256_after':h2,'stat_before':before,'stat_after':after,'stable':before==after and h==h2,'bytes':p.stat().st_size,'distribution_exact':data==old[n]}
 (OUT/'model_inventory.json').write_text(json.dumps(records,indent=2)+'\n');(OUT/'d_distribution.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'d_samples.json').write_text(json.dumps(samples,indent=2)+'\n')
 print(n, 'stable',records[n]['stable'],'distribution_exact',records[n]['distribution_exact'],flush=True)
summary={'files':len(result),'d_words':sum(x['n_blocks'] for v in result.values() for x in v.values()),'all_distributions_exact':result==old,'all_files_stable':all(x['stable'] for x in records.values()),'max_occupied_exponent':max(int(e) for v in result.values() for x in v.values() for e,f in x['frac_by_exponent_bucket'].items() if f),'elapsed_seconds':time.monotonic()-t0,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'scan_result.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)
