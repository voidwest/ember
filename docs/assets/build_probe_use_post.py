"""Build the bilingual probe/use research note from completed, sealed reports."""
from pathlib import Path
import html
import re
import runpy

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'the-probe-can-read-it'

EN = [
('p', 'My Arabic morphology work keeps coming back to a measurement problem: when a probe predicts a label from hidden states, what have I actually learned about the model? A careful split can rule out some shortcuts. It still does not establish that the model uses the decoded feature to produce its answer.'),
('p', 'The recent latent-binding experiments put that distinction under a different kind of pressure. A training-only probe could recover most of the diagnostic identities from a source representation. A learned interface into a frozen receiver transferred only one of them. Continuing the same training improved the training fit and removed that one success.'),
('callout', 'Completed diagnostic results, not robust transfer. This is a controlled letter–digit identity task, not an Arabic-language evaluation. The eight diagnostic IDs have prior evaluation and design exposure; they are not a pristine test set.'),
('h2', 'three questions that need different tests'),
('p', 'The task uses IDs made from a letter and a digit, such as M3. Training uses 56 IDs, with four existing queries per ID. The diagnostic bank contains eight whole-ID combinations absent from that positive training set. Four query variants for each identity give 32 scored cases per condition, but still only eight distinct identities.'),
('p', 'A sender supplies hidden-state features. A compressed, 64-dimensional representation carries source information to a reader that conditions a frozen receiver. The selected-ID memories used here are development fixtures: they do not demonstrate binding over a complete source table or fresh source capture.'),
('p', 'This separates three questions: can a fitted probe recover an ID from the features; can the existing receiver interface support the desired answer under supervised oracle fitting; and can a mapping trained on the 56 IDs make the receiver answer correctly for diagnostic IDs? A positive answer to one does not settle the others.'),
('h2', 'recoverable by a probe'),
('p', 'A fixed ridge probe trained on the 56 IDs decoded seven of the eight diagnostic IDs from the two sender states. The corresponding probe on the compressed representation decoded six. Neither fit used diagnostic labels or oracle codes as training targets.'),
('probe', ''),
('p', 'Leave-one-whole-ID-out checks on the training IDs were 56/56 for the two sender states and 44/56 for the compressed representation. Shifted-training-label nulls recovered 1/8 and 0/8 diagnostic IDs respectively. Independent NumPy and Torch implementations agreed within the registered numerical tolerances, with identical primary predictions.'),
('p', 'A0 was correctly decoded from sender states and missed after compression. A2 was already wrong in the sender-state probe. That describes where this particular linear decoding method loses generalization. It does not prove that compression erased the identity, or that another decoder could not recover it.'),
('h2', 'usable through the receiver'),
('p', 'The alignment experiment changed only the existing reader.value.weight matrix. It kept the sender, receiver, routing, other bridge entries, and parameter count fixed. There was no inference codebook or diagnostic-ID oracle lookup. Training used the 56 IDs and their 224 query rows for 200 fixed Adam steps.'),
('p', 'The exported matrix was loaded into the native reader. Familiar controls passed: 32/32 original cases, 32/32 donor-following cases, and 31/32 changed-binding cases. On diagnostic IDs, the result was 4/32 in each condition. All four successes belonged to M3. The other seven identities failed.'),
('p', 'Joint scoring and candidate-constrained sequential scoring agreed. This is not unrestricted generation: the sequential procedure selects among 16 prefixes and eight digits. The preregistered count and paired gates still failed. The result is limited source-driven transfer, not a successful general binding system.'),
('p', 'The probe’s 6/8 and the receiver’s one successful identity are different measurements, not a controlled causal ablation. They establish a gap between this probe’s recoverability and this mapping’s downstream behavior. They do not identify a unique failing circuit or prove that receiver-side repair is impossible.'),
('h2', 'better training fit, worse transfer'),
('p', 'A separately frozen continuation loaded the step-200 parameters and Adam moments, reproduced the starting candidate exactly, and ran 400 more identical updates. Familiar performance stayed 32/32/31. Diagnostic performance fell from 4/4/4 to 0/0/0.'),
('continuation', ''),
('p', 'The recorded pre-update training loss fell from 0.5465 to 0.0954. Teacher-forced letters were already 224/224; digits rose from 153/224 to 224/224. These are training diagnostics near the endpoints, not native evaluation scores or proof of mathematical convergence.'),
('p', 'The receiver’s held-out margins fell for six of the eight IDs between the two checkpoints. M3’s small positive margin crossed below zero. More optimization of this same objective along this trajectory did not improve transfer, even as training classification became exact.'),
('p', 'These two endpoints do not locate the best possible stopping point. Picking a new checkpoint by this diagnostic bank would use the test results for selection. Configuration choice needs training-ID-only validation. The separate four-fold work is still running at this note’s cutoff and contributes no aggregate result here.'),
('h2', 'why the negative result is interpretable'),
('p', 'The starting checkpoint replayed bitwise. Historical scores and repeated arms replayed exactly; native/render parity, direct scoring, routing, and the saved reader-state contract passed. The frozen receiver hash stayed unchanged and its state was restored. Independent score and weight reconstruction checks passed too.'),
('p', 'Those checks matter because a broken interface could produce the same headline failure. Here they support interpreting the result as poor transfer under the tested mapping and objective. They do not make the small, historically exposed diagnostic bank representative of language understanding.'),
('h2', 'back to the morphology question'),
('p', 'In morphology probing, a stricter split asks whether a decoder generalizes beyond familiar lexical material. This experiment adds another boundary: a generalizing decoder is still a decoder we trained. Its success is not evidence that the model’s own answer mechanism uses the representation in the same way.'),
('p', 'The useful connection is methodological, not a new claim about Arabic. Probe recovery, supervised reachability, and source-driven receiver behavior deserve separate evidence. Here recovery was fairly strong, native transfer was narrow, and more fitting made it worse. Keeping those results separate gives the next experiment a question it can actually answer.'),
]
AR = [
('p', 'في شغلي على الـArabic morphology، فيه سؤال يرجع كل مرة: لما الـprobe يقدر يتوقع label من الـhidden states، إيش عرفت فعلاً عن النموذج؟ تقسيم البيانات بعناية يقدر يستبعد بعض الاختصارات. بس لسه ما يثبت إن النموذج يستخدم نفس المعلومة عشان يطلع جوابه.'),
('p', 'تجارب الـlatent binding الأخيرة اختبرت الفرق هذا من جهة ثانية. Probe متدرّب على بيانات التدريب بس قدر يسترجع أغلب الـIDs التشخيصية من تمثيل المصدر. لكن الـmapping المتعلم إلى receiver ثابت نجح في ID واحد بس. ولما كملت نفس التدريب، تحسن الأداء على بيانات التدريب واختفى حتى النجاح الوحيد هذا.'),
('callout', 'هذي نتائج تشخيصية مكتملة، مو نقل قوي يعمم على IDs جديدة. المهمة مضبوطة على IDs من حرف ورقم، مو تقييم للغة العربية. والـIDs الثمانية سبق دخلت في التقييم والتصميم؛ مو test set بكر.'),
('h2', 'ثلاثة أسئلة، وكل واحد له اختباره'),
('p', 'المهمة تستخدم IDs من حرف ورقم، زي M3. التدريب على 56 ID، ولكل واحد أربعة استعلامات موجودة مسبقاً. البنك التشخيصي فيه ثمانية تركيبات ID كاملة ما دخلت ضمن مجموعة التدريب الإيجابية. أربعة استعلامات لكل ID تعطينا 32 حالة مقاسة في كل شرط، بس لسه عندنا ثماني هويات مختلفة، مو 32.'),
('p', 'الـsender يوفر features من الـhidden states. تمثيل مضغوط من 64 بُعد ينقل معلومات المصدر إلى reader يكيّف receiver ثابت. الذاكرة المجمعة للـIDs المختارة هنا تجهيزات للتطوير؛ ما تثبت binding على جدول مصدر كامل، ولا التقاط جديد من المصدر.'),
('p', 'هنا نفصل ثلاثة أسئلة: هل probe ندرّبه يقدر يسترجع الـID من الـfeatures؟ هل واجهة الـreceiver الحالية تقدر تنتج الجواب المطلوب لما نستخدم supervised oracle fitting؟ وهل mapping متدرّب على الـ56 ID يقدر يخلي الـreceiver يجاوب صح على الـIDs التشخيصية؟ نجاح واحد منها ما يحسم الباقي.'),
('h2', 'الـprobe يقدر يسترجعها'),
('p', 'Ridge probe ثابت، متدرّب على الـ56 ID، فك سبعة من الثمانية من حالتي الـsender. والـprobe المقابل على التمثيل المضغوط فك ستة. ولا واحد من التدريبين استخدم labels الـIDs التشخيصية أو oracle codes كأهداف تدريب.'),
('probe', ''),
('p', 'في فحص leave-one-whole-ID-out على بيانات التدريب، النتيجة كانت 56/56 لحالتي الـsender و44/56 للتمثيل المضغوط. ومع إزاحة labels التدريب كـnull control، النتائج التشخيصية كانت 1/8 و0/8 بالترتيب. تطبيقان مستقلان بـNumPy وTorch اتفقوا ضمن حدود الخطأ المسجلة، ونفس التوقعات الأساسية طلعت من الاثنين.'),
('p', 'A0 انقرأ صح من حالات الـsender وانفقد عند هذا الـprobe بعد الضغط. أما A2 فكان غلط من الـsender أصلاً. هذا يحدد فين طريقة القراءة الخطية هذي خسرت تعميمها. ما يثبت إن الضغط مسح الهوية، ولا إن decoder ثاني ما يقدر يسترجعها.'),
('h2', 'طيب، الـreceiver يقدر يستخدمها؟'),
('p', 'تجربة الـalignment غيرت مصفوفة reader.value.weight الموجودة بس. الـsender والـreceiver والـrouting وباقي مدخلات الجسر وعدد المعاملات كلها بقيت ثابتة. ما فيه codebook وقت الاستدلال ولا oracle lookup للـIDs التشخيصية. التدريب استخدم الـ56 ID واستعلاماتها الـ224، على 200 خطوة Adam محددة مسبقاً.'),
('p', 'المصفوفة المصدّرة انحملت في الـnative reader. الضوابط المألوفة عدّت: 32/32 للحالات الأصلية، و32/32 لاتباع المصدر المنقول، و31/32 عند تغيير الربط. في الـIDs التشخيصية، النتيجة كانت 4/32 في كل شرط. بس النجاحات الأربعة كلها كانت M3. السبعة الباقين فشلوا.'),
('p', 'الـjoint scoring والـcandidate-constrained sequential scoring اتفقوا. هذا مو توليد حر: الإجراء التسلسلي يختار من 16 بادئة وثمانية أرقام. وبوابات النجاح المسجلة مسبقاً، العددية والمقارنات المزدوجة، لسه فشلت. فيه نقل محدود من المصدر، بس ما عندنا نظام binding ناجح يعمم.'),
('p', 'نتيجة الـprobe، ستة من ثمانية، ونجاح هوية واحدة عند الـreceiver قياسان مختلفان، مو causal ablation مضبوطة. يبينوا الفجوة بين اللي الـprobe هذا يقدر يسترجعه واللي الـmapping هذا يقدر يخلي الـreceiver يسويه. ما يحددوا دائرة واحدة مسؤولة عن الفشل، ولا يثبتوا إن إصلاح جهة الـreceiver مستحيل.'),
('h2', 'تدريب أحسن، ونقل أسوأ'),
('p', 'في continuation تجمّدت لحالها، حمّلت معاملات الخطوة 200 وحالة Adam، وأعدت إنتاج نقطة البداية بالضبط، ثم كملت 400 تحديث بنفس الإعدادات. الأداء المألوف بقي 32/32/31. الأداء التشخيصي نزل من 4/4/4 إلى 0/0/0.'),
('continuation', ''),
('p', 'الـtraining loss المسجل قبل التحديث نزل من 0.5465 إلى 0.0954. مع teacher forcing، الحروف كانت أصلاً 224/224؛ والأرقام طلعت من 153/224 إلى 224/224. هذي مؤشرات تدريب قرب نقطتي النهاية، مو درجات تقييم native، ولا إثبات تقارب رياضي.'),
('p', 'هوامش الإجابة عند الـreceiver نزلت لستة من الثمانية بين النقطتين. هامش M3 الموجب الصغير صار سالب. زيادة التدريب على نفس الهدف، في المسار هذا، ما حسّنت النقل، حتى مع وصول تصنيف بيانات التدريب للدقة الكاملة.'),
('p', 'النقطتين هذي ما تحدد أفضل مكان ممكن نوقف فيه. لو اخترنا checkpoint جديد بناءً على البنك التشخيصي، نكون استخدمنا نتائج الاختبار للاختيار. اختيار الإعدادات يحتاج validation من IDs التدريب نفسها بس. شغل الـfour-fold المنفصل لسه شغال عند كتابة الملاحظة، وما فيه أي نتيجة مجمعة منه مستخدمة هنا.'),
('h2', 'ليش نقدر نفسر النتيجة السلبية؟'),
('p', 'نقطة البداية رجعت مطابقة بت ببت. الدرجات التاريخية والحالات المكررة رجعت بالضبط؛ وفحوص native/render parity والتقييم المباشر والـrouting وعقد حالة الـreader المحفوظة كلها عدّت. Hash الـreceiver الثابت ما تغير، وحالته رجعت كما كانت. فحوص إعادة بناء الأوزان والدرجات المستقلة عدّت كمان.'),
('p', 'الفحوص هذي مهمة لأن واجهة مكسورة ممكن تعطينا نفس عنوان الفشل. هنا تساعدنا نفسر النتيجة كضعف نقل تحت الـmapping والهدف اللي اختبرناهم. بس ما تخلي البنك التشخيصي الصغير، اللي سبق استخدمناه، ممثلاً لفهم اللغة عموماً.'),
('h2', 'نرجع لسؤال الـmorphology'),
('p', 'في morphology probing، التقسيم الأصعب يسأل إذا الـdecoder يعمم خارج الكلمات المألوفة. التجربة هذي تضيف حد ثاني: حتى الـdecoder اللي يعمم، لسه هو decoder إحنا درّبناه. نجاحه ما يثبت إن آلية الإجابة عند النموذج تستخدم التمثيل بنفس الطريقة.'),
('p', 'الرابط المفيد هنا في طريقة الاختبار، مو نتيجة جديدة عن العربي. استرجاع المعلومة بالـprobe، وإمكانية الوصول للجواب بتدريب oracle، وسلوك الـreceiver انطلاقاً من المصدر، كل واحد يحتاج دليله. هنا الاسترجاع كان قوي نسبياً، والنقل الفعلي محدود، وزيادة التدريب خلّته أسوأ. لما نفصل النتائج، نقدر نسأل في التجربة الجاية سؤال له جواب واضح.'),
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
    out='<figure class="probe-use-figure"><h3>'+text(title,ar)+'</h3>'
    for i,(label,n,d) in enumerate(rows):
        out+='<div class="metric"><div class="metric-label">'+text(label,ar)+f'<bdi>{n}/{d}</bdi></div><div class="metric-track"><div class="metric-bar {kind}-{i}"></div></div></div>'
    return out+'<figcaption>'+text(note,ar)+'</figcaption></figure>'


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
        body+='<h2>'+('صلة بالشغل السابق' if ar else 'related work on this site')+'</h2><ul>'
        for slug,en,arabic in [('when-the-result-gets-less-flashy-but-more-real','When the Result Gets Less Flashy but More Real','لما النتيجة تصير أقل استعراضاً وأكثر واقعية'),('before-mapping-kv-caches','Before Mapping KV Caches, Make Them Measurable','قبل ما ننقل KV caches، نخليها قابلة للقياس')]:
            body+='<li><a href="/research-notes/'+slug+suffix+'">'+text(arabic if ar else en,ar)+'</a></li>'
        body+='</ul>'+footer
        path=folder/(SLUG+suffix);path.write_text((head+'<body>\n'+body+'\n</body>\n</html>\n').replace('><', '>\n<'))
        _,new=normalizer['render_file'](path);path.write_text(new)
        print('Built',path)

if __name__=='__main__':main()
