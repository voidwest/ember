"""Render recorded post measurements as bilingual, themed PNG charts (Pillow/RAQM)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research-notes/figures'
OUT.mkdir(exist_ok=True)
FONTS={'ar':'/usr/share/fonts/TTF/Arial.TTF','en':'/usr/share/fonts/noto/NotoSans-Regular.ttf'}
DATA={
 'startup':[
  (['Model build · milliseconds','بناء النموذج · مللي ثانية'],[
   (['Cache off','بدون cache'],670,900,'670 ms'),
   (['First cache write','أول كتابة للـcache'],749,900,'749 ms'),
   (['Warm cache hit','الـcache جاهز ودافئ'],83,900,'83 ms')]),
  (['GGUF metadata parse · milliseconds','قراءة بيانات GGUF · مللي ثانية'],[
   (['Before','قبل التغيير'],37.7,48,'37.7 ms'),
   (['Validate + skip','التحقق بدون الاحتفاظ بالقيم'],10.8,48,'10.8 ms')])],
 'probe': [(['ID recovery by the probe','استرجاع الهوية بالـprobe'],[
  (['Two sender states','حالتي الـsender'],7,8,'7/8'),
  (['Compressed representation','التمثيل المضغوط'],6,8,'6/8')])],
 'continuation': [(['Training digits · teacher forcing','أرقام التدريب · teacher forcing'],[
  (['Near step 200','قرب الخطوة 200'],153,224,'153/224'),
  (['Near step 600','قرب الخطوة 600'],224,224,'224/224')]),
  (['Diagnostic cases · native original scoring','الحالات التشخيصية · التقييم الأصلي الفعلي'],[
  (['Step 200 · all successes are M3','الخطوة 200 · كل النجاحات M3'],4,32,'4/32'),
  (['Step 600','الخطوة 600'],0,32,'0/32')])]
}
NOTES={
'startup':['Independent comparisons and scales. Lower is better.','مقارنات ومقاييس مستقلة. الأقل أفضل.'],
'probe':['Eight distinct diagnostic IDs; not a native-transfer measurement.','ثماني هويات تشخيصية مختلفة؛ مو قياس نقل فعلي.'],
'continuation':['Each bar uses its own denominator. 32 cases = 8 IDs × 4 queries.','كل شريط بمقامه. 32 حالة = ثماني هويات × أربعة استعلامات.']}
for kind,panels in DATA.items():
 for lang in ['en','ar']:
  ar=lang=='ar'; idx=int(ar)
  for theme in ['light','dark']:
   bg,ink,muted,track,accent=('#ece9e1','#242424','#666763','#dedad0','#315fd6') if theme=='light' else ('#15171a','#e9e6de','#a1a19d','#292d34','#86a4f3')
   height=100+sum(110+len(rows)*130 for _,rows in panels)
   im=Image.new('RGB',(1400,height),bg);d=ImageDraw.Draw(im)
   def txt(s,y,size=30,color=ink):
    font=ImageFont.truetype(FONTS[lang],size)
    d.text((1335 if ar else 65,y),s,font=font,fill=color,anchor='ra' if ar else 'la',direction='rtl' if ar else 'ltr',language=lang)
   y=38
   for title,rows in panels:
    txt(title[idx],y,36);y+=82
    for label,value,limit,display in rows:
     txt(label[idx],y,29)
     # Values stay LTR and separate from RTL labels.
     f=ImageFont.truetype(FONTS['en'],29)
     d.text((65 if ar else 1335,y),display,font=f,fill=ink,anchor='la' if ar else 'ra')
     y+=62
     d.rectangle((65,y,1335,y+25),fill=track)
     if value:d.rectangle((65,y,65+1270*value/limit,y+25),fill=accent)
     else:d.line((65,y,65,y+25),fill=accent,width=3)
     y+=68
    y+=28
   txt(NOTES[kind][idx],height-57,23,muted)
   im.save(OUT/f'{kind}-post-{lang}-{theme}.png')
print('Rendered 12 PNG charts.')
