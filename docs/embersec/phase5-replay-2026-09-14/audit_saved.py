import hashlib,json,struct
from pathlib import Path
OUT=Path(__file__).resolve().parent
OLD=OUT.parents[2]/'docs/embersec/phase5-correction-2026-09-03'
rows=[json.loads(x) for x in (OLD/'sweep.jsonl').read_text().splitlines()]
def decode(word):
 sign=-1 if word&32768 else 1;e=(word>>10)&31;m=word&1023
 if e==31:return None
 return sign*(m*2**-24 if e==0 else (1024+m)*2**(e-25))
issues=[]
for i,r in enumerate(rows):
 b=int(r['d_bits'],16);f=int(r['faulted_bits'],16)
 if b^(1<<r['bit'])!=f:issues.append([i,'bit mutation'])
 requested=struct.unpack('<f',struct.pack('<f',r['d_requested']))[0]
 want=struct.unpack('<H',struct.pack('<e',requested))[0]
 if want!=b:issues.append([i,'RNE conversion',want,b])
 for key,word in [('d_actual',b),('faulted_d',f)]:
  val=decode(word)
  if val is not None and abs(float(r[key])-val)>max(1e-30,abs(val)*1e-9):issues.append([i,key])
eligible=[e for e in range(31) if any((e^(1<<i))==31 for i in range(5))]
assert eligible==[15,23,27,29,30]
summary={'rows':len(rows),'independent_bit_and_RNE_issues':issues,'eligible_finite_exponents':eligible,'real_rows':sum(abs(r['d_requested'])!=1 for r in rows),'real_nonfinite':sum(not r['finite'] for r in rows if abs(r['d_requested'])!=1),'control_nonfinite':sum(not r['finite'] for r in rows if abs(r['d_requested'])==1),'by_format':{}}
for d in ['Q4_K','Q6_K','Q8_0']:
 a=[r for r in rows if r['dtype']==d and abs(r['d_requested'])!=1];b=[r for r in a if r['bit']==14]
 summary['by_format'][d]={'n':len(a),'max_relative_l2':max(r['rel_l2'] for r in a),'max_absolute':max(r['max_abs'] for r in a),'bit14_argmax_changes':sum(r['top1_flipped'] for r in b),'bit14_trials':len(b)}
(OUT/'independent_audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
