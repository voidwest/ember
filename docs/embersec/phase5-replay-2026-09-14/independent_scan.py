"""Independent GGUF header walk and mmap/stride d-word histogram audit."""
from pathlib import Path
import struct,json,mmap,numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
old=json.loads((OUT/'d_distribution.json').read_text())
scalar={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
layouts={8:(32,34,0),12:(256,144,0),14:(256,210,208)}
results={}
for name,expected in old.items():
 with (ROOT/name).open('rb') as f:
  def uint(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
  def string():return f.read(uint('Q'))
  def skip(t):
   if t in scalar:f.seek(scalar[t],1)
   elif t==8:f.seek(uint('Q'),1)
   elif t==9:
    subtype=uint('I');n=uint('Q')
    if subtype in scalar:f.seek(n*scalar[subtype],1)
    else:
     for _ in range(n):skip(subtype)
   else:raise ValueError(t)
  assert f.read(4)==b'GGUF';assert uint('I')==3
  nt,nk=uint('Q'),uint('Q');alignment=32
  for _ in range(nk):
   key=string();t=uint('I')
   if key==b'general.alignment':
    assert t in (4,10);alignment=uint('I' if t==4 else 'Q')
   else:skip(t)
  desc=[]
  for _ in range(nt):
   n=string();rank=uint('I');dims=[uint('Q') for _ in range(rank)];dtype=uint('I');offset=uint('Q');desc.append((dtype,dims,offset))
  start=((f.tell()+alignment-1)//alignment)*alignment
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);hist={};ntensors={}
  for dtype,dims,offset in desc:
   if dtype not in layouts:continue
   epb,stride,d_off=layouts[dtype];nelem=1
   for d in dims:nelem*=d
   assert nelem%epb==0;n=nelem//epb
   assert start+offset+n*stride<=len(mm)
   words=np.ndarray((n,),dtype='<u2',buffer=mm,offset=start+offset+d_off,strides=(stride,))
   counts=np.bincount(words,minlength=65536).astype(np.int64);del words
   hist[dtype]=hist.get(dtype,np.zeros(65536,dtype=np.int64))+counts;ntensors[dtype]=ntensors.get(dtype,0)+1
  mm.close()
 per={}
 for dtype,h in hist.items():
  expcounts=h.reshape(2,32,1024).sum(axis=(0,2));s=expected[str(dtype)];n=int(h.sum())
  assert n==s['n_blocks'] and ntensors[dtype]==s['n_tensors']
  assert all(int(round(s['frac_by_exponent_bucket'][str(e)]*n))==int(expcounts[e]) for e in range(32))
  per[str(dtype)]={'d_words':n,'tensors':ntensors[dtype],'nonfinite':int(expcounts[31]),'single_bit_nonfinite_eligible_words':sum(int(expcounts[e]) for e in [15,23,27,29,30]),'exponent_counts':[int(c) for c in expcounts],'unique_scale_words':int(np.count_nonzero(h))}
 results[name]=per
 print(name,'PASS independent counts and exponent histogram',flush=True)
(OUT/'independent_scan.json').write_text(json.dumps(results,indent=2)+'\n')
