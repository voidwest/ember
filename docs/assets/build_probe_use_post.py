"""Build the bilingual probe/use research note from completed, sealed reports."""
from pathlib import Path
import html
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'the-probe-can-read-it'

EN = [
    ('p', 'Picture a source entry labeled M3. The task is to pass information about that entry to a second model and have it return the right letter–digit ID. The second model stays frozen; we train the interface feeding it.'),
    ('p', 'A probe recovered most IDs absent from its training set. The frozen receiver got just one identity right: M3. More training improved the training fit and erased that remaining transfer.'),
    ('callout', 'Completed results on a small letter–digit task. This is not an Arabic-language evaluation or evidence of robust transfer. The eight diagnostic IDs had prior evaluation and design exposure, so they are not a pristine test set.'),
    ('h2', 'three terms'),
    ('p', 'Probe: a separate predictor trained to read an ID from saved model features.'),
    ('p', 'Receiver: the frozen model we ask to produce the ID. A learned reader feeds source features into it.'),
    ('p', 'Transfer: the receiver answers correctly for a whole ID combination absent from the positive training set.'),
    ('h2', 'the task'),
    ('p', 'We train on familiar IDs and check whole letter–digit combinations left out of that training set.'),
    ('p', 'Training contains 56 IDs with four queries each: 224 rows. The diagnostic bank has eight IDs and four queries each: 32 cases per condition. Those are eight distinct identities, not 32 independent identities.'),
    ('p', 'The sender supplies hidden-state features through a compressed, 64-dimensional representation. The selected-ID memories are development fixtures; full-table binding and fresh source capture remain untested.'),
    ('p', 'Reading an ID with a probe, reaching it with supervised oracle fitting, and transferring it through the receiver are separate tests. Each needs its own evidence.'),
    ('h2', 'the probe recovered most IDs'),
    ('p', 'The fixed ridge probe generalized to most diagnostic identities.'),
    ('p', 'It recovered 7/8 from the two sender states and 6/8 from the compressed representation. Both fits used only the 56 training IDs, without diagnostic labels or oracle-code targets.'),
    ('probe', ''),
    ('p', 'Leaving one whole training ID out at a time gave 56/56 for sender states and 44/56 after compression. Shifting the training labels as a null control gave diagnostic scores of 1/8 and 0/8, respectively.'),
    ('p', 'Independent NumPy and Torch implementations agreed within the registered tolerances and produced identical primary predictions.'),
    ('p', 'A0 was read correctly before compression and missed afterward. A2 was already wrong in the sender-state probe. This locates errors for this decoder; it does not establish that compression erased identity information.'),
    ('h2', 'the receiver transferred one ID'),
    ('p', 'The trained interface made the frozen receiver answer correctly for M3. The other seven diagnostic identities failed.'),
    ('p', 'That is 4/32 cases in each of the original, donor-following, and changed-binding conditions. All four successful queries belonged to M3.'),
    ('p', 'Familiar controls passed: 32/32 original cases, 32/32 donor-following cases, and 31/32 changed-binding cases.'),
    ('p', 'Training changed only reader.value.weight for 200 fixed Adam steps on the 224 training rows. The sender, receiver, routing, other bridge entries, and parameter count stayed fixed. There was no inference codebook or diagnostic-ID lookup.'),
    ('p', 'Joint and candidate-constrained sequential scoring agreed. Sequential scoring chooses among 16 prefixes and eight digits; this is not unrestricted generation. The preregistered count and paired gates failed.'),
    ('p', 'Probe recovery of 6/8 and receiver success on one ID measure different operations. Their gap leaves the failing mechanism unresolved.'),
    ('h2', 'more training erased the transfer'),
    ('p', 'Continuing the same training removed the receiver’s only diagnostic success.'),
    ('p', 'The continuation replayed the step-200 parameters and Adam state exactly, then ran 400 more updates. Diagnostic scores fell from 4/4/4 to 0/0/0 across the three conditions. Familiar scores stayed at 32/32/31.'),
    ('continuation', ''),
    ('p', 'Recorded pre-update training loss fell from 0.5465 to 0.0954. With teacher forcing, digit accuracy rose from 153/224 to 224/224. Letter accuracy was already 224/224.'),
    ('p', 'These training measurements are taken near the endpoints. They neither measure native transfer nor establish mathematical convergence.'),
    ('p', 'Receiver margins fell for six of the eight diagnostic IDs. M3’s small positive margin turned negative. Along this trajectory, exact training classification came with worse transfer.'),
    ('p', 'Choosing another checkpoint from these diagnostic scores would use test results for selection. Step choice needs training-ID-only validation. The separate four-fold work had no aggregate result at this note’s cutoff.'),
    ('h2', 'the checks passed'),
    ('p', 'The replay and interface checks passed, supporting the interpretation of poor transfer under this mapping.'),
    ('p', 'The starting checkpoint, historical scores, and repeated arms replayed exactly. Native/render parity, direct scoring, routing, and the saved reader-state contract passed. Independent score and weight reconstruction checks also passed.'),
    ('p', 'The frozen receiver hash stayed unchanged and its state was restored. These checks establish the tested mechanics; the eight exposed diagnostic IDs still cannot represent language understanding broadly.'),
    ('h2', 'what this adds to the morphology work'),
    ('p', 'The morphology notes ask whether a probe generalizes beyond familiar words. This task adds a separate receiver test: six readable compressed IDs yielded one transferred ID.'),
    ('p', 'The practical consequence is to report probe recovery and model behavior separately. A claim about the model using a feature needs a behavioral intervention. This toy task provides no new Arabic-language result.'),
]
AR = [
    ('p', 'تخيّل عندك معلومة في المصدر باسم M3. المطلوب نمرّر معلوماتها لنموذج ثاني، ونخليه يرجّع الـID الصحيح: الحرف والرقم. النموذج الثاني ثابت؛ اللي ندرّبه هو الواجهة اللي توصّل له المعلومة.'),
    ('p', 'الـprobe قدر يقرأ أغلب الـIDs اللي ما شافها في التدريب. بس لما مرّرنا المعلومة للـreceiver الثابت، نجح ID واحد بس: M3. وكملنا التدريب، فتحسن الأداء على التدريب واختفى النقل.'),
    ('callout', 'هذي نتائج مكتملة على مهمة صغيرة من حرف ورقم. مو تقييم للغة العربية، ولا دليل على نقل قوي يعمم. والـIDs الثمانية سبق دخلت في التقييم والتصميم؛ مو test set بكر.'),
    ('h2', 'ثلاث كلمات قبل الأرقام'),
    ('p', 'Probe: متنبئ منفصل ندرّبه يقرأ الـID من features محفوظة من النموذج.'),
    ('p', 'Receiver: النموذج الثابت اللي نطلب منه يطلع الـID. نوصل له features المصدر عن طريق reader ندرّبه.'),
    ('p', 'النقل: الـreceiver يجاوب صح على تركيب حرف ورقم كامل ما دخل في مجموعة التدريب الإيجابية.'),
    ('h2', 'إيش المهمة؟'),
    ('p', 'ندرّب على IDs مألوفة، وبعدها نختبر تركيبات كاملة من حرف ورقم تركناها خارج التدريب.'),
    ('p', 'التدريب فيه 56 ID، ولكل واحد أربعة استعلامات. المجموع 224 صف تدريب.'),
    ('p', 'البنك التشخيصي فيه ثمانية IDs، ولكل واحد أربعة استعلامات. يعني 32 حالة في كل شرط، بس ثماني هويات مختلفة، مو 32 هوية مستقلة.'),
    ('p', 'الـsender يوفر features من الـhidden states، ونمرّرها في تمثيل مضغوط من 64 بُعد. ذاكرة الـIDs المختارة هنا تجهيز للتطوير. لسه ما اختبرنا الربط على جدول مصدر كامل أو التقاط جديد من المصدر.'),
    ('p', 'قراءة الـID بالـprobe، والوصول له بتدريب oracle مباشر، ونقله للـreceiver ثلاثة اختبارات منفصلة. كل واحد يحتاج دليله.'),
    ('h2', 'الـprobe قرأ أغلب الـIDs'),
    ('p', 'الـridge probe الثابت عمّم على أغلب الهويات التشخيصية.'),
    ('p', 'قرأ 7/8 من حالتي الـsender، و6/8 من التمثيل المضغوط. الاثنين تدربوا على الـ56 ID بس. ما استخدمنا labels تشخيصية أو oracle codes كأهداف تدريب.'),
    ('probe', ''),
    ('p', 'لما نترك ID كامل من التدريب كل مرة ونختبره، النتيجة 56/56 من حالات الـsender. بعد الضغط صارت 44/56.'),
    ('p', 'ومع إزاحة labels التدريب كضابط null، النتيجة التشخيصية صارت 1/8 و0/8 بالترتيب.'),
    ('p', 'تطبيقان مستقلان بـNumPy وTorch اتفقوا ضمن حدود الخطأ المسجلة. التوقعات الأساسية كانت نفسها.'),
    ('p', 'A0 انقرأ صح قبل الضغط، وغلط بعده. A2 كان غلط من الـsender أصلاً. هذا يحدد أخطاء القارئ هذا؛ ما يثبت إن الضغط مسح معلومات الهوية.'),
    ('h2', 'الـreceiver نقل ID واحد'),
    ('p', 'الواجهة المتدرّبة خلت الـreceiver الثابت يجاوب صح على M3. الهويات التشخيصية السبعة الباقية فشلت.'),
    ('p', 'النتيجة 4/32 في كل شرط: الأصلي، واتباع المصدر المنقول، وتغيير الربط. الاستعلامات الأربعة الناجحة كلها تخص M3.'),
    ('p', 'الضوابط المألوفة عدّت. الأصلي جاب 32/32، واتباع المصدر المنقول 32/32، وتغيير الربط 31/32.'),
    ('p', 'غيّرنا reader.value.weight بس. التدريب كان 200 خطوة Adam محددة مسبقاً، على الـ224 صف تدريب.'),
    ('p', 'الـsender والـreceiver والـrouting وباقي مدخلات الجسر وعدد المعاملات بقيت ثابتة. ما فيه codebook وقت الاستدلال أو بحث عن جواب الـID التشخيصي.'),
    ('p', 'التقييم المشترك والتقييم التسلسلي المقيد اتفقوا. التسلسلي يختار من 16 بادئة وثمانية أرقام؛ مو توليد حر. بوابات النجاح العددية والمقارنات المزدوجة المسجلة مسبقاً فشلت.'),
    ('p', 'الـprobe قرأ 6/8، والـreceiver نجح في هوية واحدة. هذي عمليتان مختلفتان، والفجوة بينهم ما تحدد آلية الفشل.'),
    ('h2', 'زيادة التدريب شالت النقل'),
    ('p', 'لما كملنا نفس التدريب، اختفى النجاح التشخيصي الوحيد عند الـreceiver.'),
    ('p', 'أعدنا معاملات الخطوة 200 وحالة Adam بالضبط، ثم كملنا 400 تحديث. النتائج التشخيصية نزلت من 4/4/4 إلى 0/0/0 في الشروط الثلاثة. النتائج المألوفة بقيت 32/32/31.'),
    ('continuation', ''),
    ('p', 'الـtraining loss المسجل قبل التحديث نزل من 0.5465 إلى 0.0954.'),
    ('p', 'مع teacher forcing، دقة الأرقام طلعت من 153/224 إلى 224/224. الحروف كانت أصلاً 224/224.'),
    ('p', 'هذي قياسات تدريب قرب نقطتي النهاية. ما تقيس النقل الفعلي، ولا تثبت تقارب رياضي.'),
    ('p', 'هوامش إجابة الـreceiver نزلت لستة من الثمانية. هامش M3 الموجب الصغير صار سالب. في المسار هذا، دقة التدريب الكاملة جات مع نقل أسوأ.'),
    ('p', 'لو اخترنا checkpoint ثاني من الدرجات التشخيصية، نكون استخدمنا الاختبار للاختيار. اختيار الخطوة يحتاج validation من IDs التدريب بس. شغل الـfour-fold المنفصل ما كان عنده نتيجة مجمعة عند تاريخ الملاحظة.'),
    ('h2', 'الفحوص عدّت'),
    ('p', 'فحوص الإعادة والواجهة عدّت. هذا يدعم قراءة النتيجة كضعف نقل تحت الـmapping اللي اختبرناه.'),
    ('p', 'نقطة البداية والدرجات التاريخية والحالات المكررة رجعت مطابقة. فحوص native/render parity والتقييم المباشر والـrouting وعقد حالة الـreader المحفوظة عدّت.'),
    ('p', 'فحوص إعادة بناء الأوزان والدرجات المستقلة عدّت كمان. Hash الـreceiver ما تغير، وحالته رجعت كما كانت.'),
    ('p', 'الفحوص تثبت آلية التشغيل المختبرة. بس الثمانية IDs اللي سبق استخدمناها ما تمثل فهم اللغة عموماً.'),
    ('h2', 'إيش يضيف لشغل الصرف؟'),
    ('p', 'ملاحظات الصرف تسأل إذا الـprobe يعمم خارج الكلمات المألوفة. هنا أضفنا اختبار للـreceiver: ستة IDs تنقرأ بعد الضغط، وID واحد ينتقل.'),
    ('p', 'عشان كذا نفصل نتيجة الـprobe عن سلوك النموذج في التقرير. ادعاء إن النموذج يستخدم المعلومة يحتاج تدخل سلوكي. المهمة الصغيرة هذي ما تضيف نتيجة جديدة عن اللغة العربية.'),
]

REPORTS = [
('latent_binding_training_feature_probe_20260911', 'Training-only wire probe', 'Probe على التمثيل المضغوط'),
('latent_binding_sender_feature_probe_ties_20260911', 'Sender-state comparison', 'مقارنة حالات الـsender'),
('latent_binding_training_alignment_20260911', 'Native training-only alignment', 'الـalignment الفعلي من بيانات التدريب'),
('latent_binding_convergence_fixed_20260911', 'Fixed continuation', 'تكملة التدريب بنفس الإعدادات'),
]

def text(s, ar=False):
    s=html.escape(s)
    if ar:
        # Isolate Latin terms and numbers in RTL prose, preserving HTML entities.
        s=re.sub(r'&[^;]+;|[A-Za-z0-9][A-Za-z0-9_./–-]*(?: [A-Za-z0-9][A-Za-z0-9_./–-]*)*',lambda m:m[0] if m[0].startswith('&') else '<bdi>'+m[0]+'</bdi>',s)
    return s


def figure(kind, ar):
    if kind=='probe':
        title='استرجاع الـID بالـprobe' if ar else 'ID recovery by the probe'
        rows=[('حالتي الـsender' if ar else 'Two sender states',7,8),('التمثيل المضغوط' if ar else 'Compressed representation',6,8)]
        note='نفس البنك التشخيصي: ثماني هويات مختلفة. مقارنة probe، مو قياس نقل فعلي.' if ar else 'Same diagnostic bank: eight distinct IDs. Probe comparison, not a native-transfer measurement.'
    else:
        title='نقطتان على نفس مسار التدريب' if ar else 'Two endpoints on the same training trajectory'
        rows=[('أرقام التدريب · قرب 200' if ar else 'Training digits · near 200',153,224),('أرقام التدريب · قرب 600' if ar else 'Training digits · near 600',224,224),('حالات تشخيصية · 200' if ar else 'Diagnostic cases · 200',4,32),('حالات تشخيصية · 600' if ar else 'Diagnostic cases · 600',0,32)]
        note='الأشرطة نسب داخل كل مقياس. التدريب: teacher forcing قبل التحديث. التشخيص: original native scoring، ثماني هويات × أربعة استعلامات. كل النجاحات تخص M3.' if ar else 'Bars are proportions within each metric. Training: pre-update teacher forcing. Diagnostic: original native scoring, eight IDs × four queries. All successes belong to M3.'
    lang = 'ar' if ar else 'en'
    out = '<figure>'
    for theme in ('dark', 'light'):
        out += '<img class="chart theme-chart-' + theme + '" src="figures/' + kind + '-post-' + lang + '-' + theme + '.png" alt="' + html.escape(title + ': ' + '; '.join(label + ' ' + str(n) + '/' + str(d) for label,n,d in rows), quote=True) + '">'
    return out + '<figcaption>' + text(note,ar) + '</figcaption></figure>'



def main():
    folder=ROOT/'research-notes'
    css='''/* Layout only: chart colors follow the shared site theme. */
.probe-use-figure{padding:20px;direction:inherit}
.probe-use-figure h3{margin:0 0 24px}
.probe-use-figure .metric{margin:22px 0}
.probe-use-figure .metric-label{display:flex;justify-content:space-between;gap:12px;font-size:14px;margin-bottom:8px}
.probe-use-figure .metric-label>bdi{white-space:nowrap;font-family:var(--mono-font)}
.probe-use-figure .metric-track{height:20px;background:var(--surface-2);direction:ltr;border-inline-start:1px solid var(--border)}
.probe-use-figure .metric-bar{height:100%;background:var(--accent)}
.probe-use-figure figcaption{color:var(--muted);font-size:13px;margin-top:22px}
.probe-use-figure .continuation-0,.probe-use-figure .continuation-1{background:var(--muted)}
'''
    for key,val in [('probe-0',7/8),('probe-1',6/8),('continuation-0',153/224),('continuation-1',1),('continuation-2',4/32),('continuation-3',0)]:
        css+=f'.probe-use-figure .{key}'+'{width:'+str(val*100)+'%}\n'
    (folder/(SLUG+'.css')).write_text(css)
    normalizer=runpy.run_path(str(ROOT.parent/'scripts/build_docs.py'))
    for ar,sections in [(False,EN),(True,AR)]:
        suffix='.ar.html' if ar else '.html'
        template=(folder/('before-mapping-kv-caches'+suffix)).read_text()
        title='الـprobe يقدر يقرأها، بس النموذج يقدر يستخدمها؟' if ar else 'the probe can read it. can the model use it?'
        head=template.split('<body>')[0]
        head=re.sub(r'<title>.*?</title>','<title>'+title+'</title>',head)
        for prop,value in [('og:title',title),('og:description','استرجاع الهوية بالـprobe مقابل استخدامها عند الـreceiver: نتائج تشخيصية مكتملة وحدودها.' if ar else 'Probe recovery and native transfer in a controlled identity task: completed diagnostic results and their limits.'),('og:url','https://voidwest.dev/research-notes/'+SLUG+suffix)]:
            head=re.sub(r'(<meta property="'+prop+r'" content=")[^"]*(")',lambda m:m[1]+html.escape(value,quote=True)+m[2],head)
        head=head.replace('</head>','<link rel="stylesheet" href="'+SLUG+'.css">\n</head>')
        nav=re.search(r'<!-- docs:nav start -->.*?<!-- docs:nav end -->',template,re.S)[0].replace('before-mapping-kv-caches',SLUG)
        back=re.search(r'<div class="back">.*?</div>',template,re.S)[0]
        footer=re.search(r'<!-- docs:footer start -->.*?<!-- docs:footer end -->',template,re.S)[0]
        meta='2026-09-13 · تجارب مكتملة من 11 سبتمبر · بنك تشخيصي من ثمانية IDs' if ar else '2026-09-13 · completed September 11 experiments · eight-ID diagnostic bank'
        body=nav+back+'<h1>'+text(title,ar)+'</h1><div class="meta">'+text(meta,ar)+'</div>'
        for kind,content in sections:
            if kind in ('probe','continuation'):body+=figure(kind,ar)
            elif kind=='callout':body+='<div class="callout observation"><div class="label">'+('حدود النتيجة' if ar else 'claim boundary')+'</div><p>'+text(content,ar)+'</p></div>'
            else:body+='<'+kind+'>'+text(content,ar)+'</'+kind+'>'
            if kind == 'callout':
                body += '<p><a href="https://github.com/voidwest/ember#five-minute-workflow">' + text('مسار الالتقاط والتدخل وحفظ الـbundles في Ember' if ar else "Ember's capture, intervention, and bundle workflow", ar) + '</a>' + text(' يساعدنا نفحص تجارب استخدام المعلومة ونعيد تشغيلها.' if ar else ' supports inspecting and replaying tests of feature use.', ar) + '</p>'

        body+='<h2>'+('صلة بالشغل السابق' if ar else 'related work on this site')+'</h2><ul>'
        for slug,en,arabic in [('when-the-result-gets-less-flashy-but-more-real','When the Result Gets Less Flashy but More Real','لما النتيجة تصير أقل استعراضاً وأكثر واقعية'),('before-mapping-kv-caches','Before Mapping KV Caches, Make Them Measurable','قبل ما ننقل KV caches، نخليها قابلة للقياس')]:
            body+='<li><a href="/research-notes/'+slug+suffix+'">'+text(arabic if ar else en,ar)+'</a>'+text((' — يشرح كيف التقسيم ومكان قراءة الحالة يغيروا معنى درجة الـprobe.' if slug.startswith('when-') else ' — يشرح فحوص الواجهة وإعادة التشغيل اللي نحتاجها قبل اختبار النقل.') if ar else (' — explains how splits and readout position change the meaning of a probe score.' if slug.startswith('when-') else ' — describes the interface and replay checks needed before a transfer test.'),ar)+'</li>'
        body+='</ul>'+footer
        path=folder/(SLUG+suffix);path.write_text((head+'<body>\n'+body+'\n</body>\n</html>\n').replace('><', '>\n<'))
        _,new=normalizer['render_file'](path);path.write_text(new)
        print('Built',path)

if __name__=='__main__':main()
