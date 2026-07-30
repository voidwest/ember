(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))s(a);new MutationObserver(a=>{for(const r of a)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&s(o)}).observe(document,{childList:!0,subtree:!0});function n(a){const r={};return a.integrity&&(r.integrity=a.integrity),a.referrerPolicy&&(r.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?r.credentials="include":a.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function s(a){if(a.ep)return;a.ep=!0;const r=n(a);fetch(a.href,r)}})();const F={ar:{ui:{skip:"تجاوز إلى المحتوى",theme:"تبديل المظهر",language:"English",menu:"الفهرس",close:"إغلاق الفهرس",source:"الورقة الأصلية",reset:"إعادة الضبط",reported:"بيانات الورقة",simulation:"محاكاة تعليمية",editorial:"قراءة تحريرية",higher:"الأعلى أفضل",lower:"الأقل أفضل"},hero:{eyebrow:"PAPER NOTE / ARCHITECTURE / 2017",dek:"كيف صار attention هو مسار الحساب نفسه، بدل ما يكون وصلة فوق RNN.",contribution:"الـ Transformer يبقي بنية encoder–decoder، لكنه يشيل recurrence وconvolution من طبقاتها الأساسية. كل موضع يقرأ المواضع الأخرى عبر attention، والترتيب يدخل بإشارة صريحة.",reading:"قراءة تقنية · 28 دقيقة",prerequisites:"جبر خطي · softmax · encoder–decoder",disclosure:"هذه النسخة العربية مكتوبة من معنى الورقة ومحرّرة تقنياً. مو ترجمة آلية ولا ترجمة سطر مقابل سطر.",openPaper:"افتح PDF",begin:"ابدأ من السؤال",resultLabel:"WMT14 EN→DE",resultNote:"Transformer (big) · Table 2",secondLabel:"WMT14 EN→FR",secondNote:"رقم الجدول والملخص؛ §6.1 يكتب 41.0"},nav:{overview:"السؤال",comparison:"قبل Transformer",architecture:"المعمارية",attention:"معادلة attention",position:"الموضع والتعقيد",training:"التدريب",results:"النتائج",boundaries:"حدود الادعاء",glossary:"المصطلحات"},overview:{kicker:"01 / السؤال",title:"المشكلة مو إن النماذج ما كانت تنتبه",lead:"الـ attention كان موجوداً قبل هذه الورقة. العقدة أن recurrent models كانت لسه تمرّ على التسلسل خطوة خطوة، وconvolution تحتاج عمقاً إضافياً عشان تربط موضعين بعيدين.",body:"السؤال الأضيق: هل نقدر نبني encoder وdecoder من attention وfeed-forward layers فقط؟ الورقة تقول نعم، ثم تختبر هذا التصميم على ترجمة EN–DE وEN–FR وعلى constituency parsing.",claimTitle:"ما تدعمه الورقة",claim:"التصميم حقق 28.4 BLEU على WMT 2014 EN→DE، أعلى بـ2.04 من أفضل صف سابق في Table 2، مع parallelism أكبر أثناء التدريب.",boundaryTitle:"ما لا تدعمه",boundary:"هذه مو مقارنة شاملة لكل مهمة أو طول تسلسل أو نظام تشغيل. والـ attention maps في الملحق ملاحظة نوعية، مو برهان سببي على interpretability.",mechanismTitle:"الفكرة في سطر",mechanism:"كل query تقارن نفسها بالمفاتيح، تحوّل الدرجات إلى أوزان، ثم تجمع values. Multi-head attention يكرر المسار بإسقاطات متعلّمة مختلفة."},comparison:{kicker:"02 / قبل Transformer",title:"ثلاث كلف، وثلاثة أطوال للمسار",intro:"Table 1 تفصل بين كلفة الطبقة، عدد الخطوات المتسلسلة، وأطول مسار بين موضعين. خلط هذه المقاييس يعطي قراءة غلط.",layer:"نوع الطبقة",cost:"كلفة الطبقة",sequential:"خطوات متسلسلة",path:"أطول مسار",note:"عندما n < d، الحد O(n²d) أصغر من O(nd²). هذا تحليل asymptotic، مو benchmark للـ latency ولا وعد أن كل kernel أسرع."},architecture:{kicker:"03 / المعمارية",title:"نفس الهيكل العام، مسار داخلي مختلف",intro:"الورقة تصف إعداد base في المتن: 6 طبقات، d_model=512، و8 heads. إعداد big الذي حقق الرقم الرئيسي أوسع ويستخدم 16 heads. بدّل بينهما عشان ما تختلط النتيجة بالمعمارية.",base:"Base",big:"Big",encoder:"Encoder × 6",decoder:"Decoder × 6",input:"embedding + positional encoding",masked:"masked self-attention",selfAttention:"multi-head self-attention",crossAttention:"encoder–decoder attention",feedForward:"position-wise feed-forward",output:"linear + softmax",addNorm:"residual + LayerNorm",modelWidth:"d_model",innerWidth:"d_ff",heads:"heads",keyWidth:"d_k = d_v",parameters:"parameters",steps:"train steps",dropout:"P_drop",smoothing:"ε_ls",stageLabel:"افحص المرحلة",stageCopy:["تُجمع positional encoding مع embeddings في أسفل encoder وdecoder. الأبعاد تبقى d_model.","في self-attention تأتي Q وK وV من خرج الطبقة السابقة نفسها. كل head يستخدم إسقاطاته المتعلّمة.","الـ decoder يمنع الموضع من رؤية التوكنات اللاحقة بوضع −∞ قبل softmax، مع إزاحة output embeddings موضعاً واحداً.","في encoder–decoder attention تأتي Q من decoder، بينما K وV تأتيان من خرج encoder.","كل موضع يمر مستقلاً عبر تحويلين خطيين وبينهما ReLU: d_model → d_ff → d_model.","كل sub-layer محاطة بـ residual connection ثم LayerNorm. هذه الخطوات موجودة بعد كل attention وFFN، مو مرة واحدة لكل block."],stages:["embedding + position","self-attention","future mask","cross-attention","feed-forward","Add & Norm"]},attention:{kicker:"04 / EQUATION 1 · محاكاة تعليمية",title:"الـ scaling مو تفصيل تجميلي",intro:"عندما يكبر d_k، يكبر تباين dot products. القسمة على √d_k تقلل دخول softmax في مناطق مشبعة ذات gradients صغيرة.",objective:"غيّر d_k. المتجهات هنا لها فعلاً البُعد المعروض، والقاسم المستخدم ظاهر بجانب النتيجة.",dimension:"بُعد المفتاح d_k",raw:"raw score",scaled:"scaled score",weight:"softmax weight",output:"الناتج الموزون",divisor:"القاسم الفعلي",inputNote:"Q وK وV قيم حتمية للتعليم، مو activations مسحوبة من النموذج. المقارنة تشرح Eq. 1 ولا تعيد تجربة الورقة."},position:{kicker:"05 / الموضع والأشكال",title:"الترتيب يدخل من خارج attention",intro:"لأن النموذج ما فيه recurrence أو convolution، تُجمع positional encoding مع embeddings. كل زوج أبعاد يشترك في التردد نفسه: sin عند 2i وcos عند 2i+1.",position:"الموضع pos",pairNote:"d0/d1 زوج، d2/d3 زوج، وهكذا.",shapesTitle:"تتبّع الأشكال",sequence:"طول التسلسل n",beforeProjection:"قبل الإسقاط",perHead:"داخل كل head",concatenated:"بعد concat",shapeNote:"في الإعدادين base وbig يبقى d_k=d_v=64؛ الذي يتغير هو عدد heads وعرض concat.",complexityTitle:"مقارنة الحدود",complexityIntro:"هذه الحدود من Table 1 بعد حذف الثوابت. نعرضها كعدد نسبي للعمليات، مو كزمن تنفيذ.",kernel:"kernel width k",neighborhood:"restricted window r",crossoverBefore:"عند هذا الطول، حد self-attention أصغر من حد recurrent.",crossoverAfter:"عند n ≥ d_model، يلحق الحد التربيعي بالـ recurrent أو يتجاوزه."},training:{kicker:"06 / التدريب",title:"الإعداد جزء من النتيجة",intro:"الـ BLEU مو رقم منفصل عن البيانات، checkpoint averaging، وdecode. هذه التفاصيل هي أداة القياس.",facts:[["WMT14 EN–DE","≈4.5M pairs · shared BPE ≈37K"],["WMT14 EN–FR","36M pairs · word-piece 32K"],["Batch","≈25K source + ≈25K target tokens"],["Hardware","1 machine · 8 × NVIDIA P100"],["Adam","β₁=.9 · β₂=.98 · ε=10⁻⁹"],["Decode","beam=4 · α=.6 · max=input+50"]],scheduleTitle:"Eq. 3 / learning-rate schedule",scheduleIntro:"يرتفع lrate خطياً لأول 4000 خطوة، ثم ينخفض بعكس الجذر. d_model يغيّر المقياس كله.",step:"step",rate:"learning rate",warmup:"warmup=4000",checkpoint:"نتائج base تستخدم متوسط آخر 5 checkpoints؛ big يستخدم آخر 20. Ablations في Table 3 لا تستخدم checkpoint averaging. نسخة big لـEN–FR استخدمت P_drop=0.1 بدل 0.3."},results:{kicker:"07 / النتائج",title:"الرقم الرئيسي واضح. مصدره يحتاج دقة.",intro:"Table 2 تقارن BLEU على newstest2014 وكلفة التدريب المقدّرة بـFLOPs. اختر المهمة؛ كل الصفوف المبلّغ عنها موجودة هنا.",german:"EN→DE",french:"EN→FR",model:"Model",type:"Type",bleu:"BLEU",cost:"Training cost",single:"single",ensemble:"ensemble",transformer:"Transformer",missing:"لم يُبلّغ",discrepancyTitle:"الورقة نفسها غير متسقة هنا",discrepancy:"الملخص وTable 2 يكتبان 41.8 لـEN→FR. نص §6.1 يكتب 41.0. نعرض 41.8 لأنه رقم الجدول والملخص، ونبقي التعارض ظاهراً بدل ما نختار بصمت.",ablationTitle:"عدد heads: مو كل زيادة أفضل",ablationIntro:"Rows A تغيّر h وd_k=d_v مع إبقاء الكلفة التقريبية ثابتة. القياس على newstest2013، بلا checkpoint averaging.",heads:"heads",devBleu:"dev BLEU",devPpl:"dev PPL",finding:"رأس واحد أقل بـ0.9 BLEU من أفضل قيمة. لكن 32 heads تهبط إلى 25.4؛ النتيجة تدعم multi-head، مو قاعدة «الأكثر دائماً أفضل».",otherTitle:"ما تقوله بقية ablations",parsingTitle:"التعميم خارج الترجمة",parsing:"على WSJ Section 23، Transformer من 4 طبقات حقق 91.3 F1 بتدريب WSJ فقط و92.7 في الإعداد semi-supervised. وصل لنتيجة قوية، لكنه ما تجاوز أفضل صف في الجدول: 93.3 لـRNN Grammar.",training:"Training",f1:"WSJ 23 F1"},boundaries:{kicker:"08 / حدود الادعاء",title:"ما الذي أثبتته الورقة فعلاً؟",lead:"أثبتت جدوى تصميم بلا recurrence أو convolution في طبقات sequence transduction الأساسية. المساحة التجريبية لسه محدودة.",items:[["مذكور في الورقة","self-attention كامل له كلفة O(n²d). الورقة تقترح restricted attention للتسلسلات الطويلة، لكنها تترك اختباره لعمل لاحق."],["مذكور في الورقة","التوليد في decoder بقي autoregressive ومتسلسلاً؛ الخاتمة تسمي تقليل هذه التسلسلية هدفاً لاحقاً."],["قراءة تحريرية","attention maps في الملحق تبدو مرتبطة ببنية نحوية ودلالية، لكنها أمثلة مختارة وليست اختباراً سببياً أو مقياس interpretability."],["قراءة تحريرية","النتائج تغطي مهمتي ترجمة وconstituency parsing. ما تكفي لتعميم التفوق على كل modality أو طول أو مهمة."]],closing:"الورقة غيّرت مسار الحساب، وبياناتها تدعم هذا التغيير. الادعاء الأقوى من كذا يحتاج تجارب ما كانت موجودة هنا."},glossary:{kicker:"09 / المصطلحات",title:"اختيارات اللغة",intro:"نستخدم العربية عندما تكون دقيقة وطبيعية، ونبقي مصطلحات ML بالإنجليزية عندما تكون هي لغة العمل الفعلية.",term:"المصطلح",definition:"التعريف",editorial:"ملاحظة تحريرية"},footer:{title:"Source",citation:"Vaswani et al., “Attention Is All You Need,” NIPS 2017, arXiv:1706.03762v7.",note:"كل النتائج والجداول هنا من الـPDF المحلي. الحسابات التفاعلية موسومة بوضوح بوصفها محاكاة تعليمية.",top:"ارجع للأعلى"}},en:{ui:{skip:"Skip to content",theme:"Toggle theme",language:"العربية",menu:"Contents",close:"Close contents",source:"Source paper",reset:"Reset",reported:"Paper data",simulation:"Teaching simulation",editorial:"Editorial analysis",higher:"Higher is better",lower:"Lower is better"},hero:{eyebrow:"PAPER NOTE / ARCHITECTURE / 2017",dek:"How attention became the computational path itself instead of an attachment to an RNN.",contribution:"The Transformer retains an encoder–decoder structure while removing recurrence and convolution from its core layers. Positions read one another through attention; order enters through an explicit signal.",reading:"Technical reading · 28 minutes",prerequisites:"Linear algebra · softmax · encoder–decoder",disclosure:"The Arabic edition is independently authored from the paper’s meaning and technically edited; it is neither machine translated nor sentence aligned.",openPaper:"Open PDF",begin:"Start with the question",resultLabel:"WMT14 EN→DE",resultNote:"Transformer (big) · Table 2",secondLabel:"WMT14 EN→FR",secondNote:"Table/abstract value; §6.1 says 41.0"},nav:{overview:"The question",comparison:"Before Transformer",architecture:"Architecture",attention:"Attention equation",position:"Position and cost",training:"Training",results:"Results",boundaries:"Claim boundaries",glossary:"Terminology"},overview:{kicker:"01 / THE QUESTION",title:"The problem was not an absence of attention",lead:"Attention predated this paper. The bottleneck was that recurrent models still traversed sequences step by step, while convolution needed more depth to connect distant positions.",body:"The narrower question is whether an encoder and decoder can be built from attention and feed-forward layers alone. The paper answers yes, then tests the design on EN–DE and EN–FR translation and constituency parsing.",claimTitle:"What the paper supports",claim:"The design reached 28.4 BLEU on WMT 2014 EN→DE, 2.04 above the strongest prior row in Table 2, with greater training parallelism.",boundaryTitle:"What it does not support",boundary:"This is not a comprehensive comparison across tasks, sequence lengths, or runtimes. The appendix attention maps are qualitative observations, not causal proof of interpretability.",mechanismTitle:"The idea in one line",mechanism:"Each query is compared with keys, the scores become weights, and values are summed. Multi-head attention repeats that path through distinct learned projections."},comparison:{kicker:"02 / BEFORE TRANSFORMER",title:"Three costs and three path lengths",intro:"Table 1 separates per-layer complexity, sequential operations, and maximum path length. Treating them as one metric produces the wrong reading.",layer:"Layer type",cost:"Per-layer cost",sequential:"Sequential ops",path:"Maximum path",note:"When n < d, O(n²d) is smaller than O(nd²). This is asymptotic analysis—not a latency benchmark or a promise that every kernel is faster."},architecture:{kicker:"03 / ARCHITECTURE",title:"The outer structure stays; the internal path changes",intro:"The paper describes the base setup in its main text: 6 layers, d_model=512, and 8 heads. The big model behind the headline result is wider and uses 16 heads. Toggle them to keep result and configuration distinct.",base:"Base",big:"Big",encoder:"Encoder × 6",decoder:"Decoder × 6",input:"embedding + positional encoding",masked:"masked self-attention",selfAttention:"multi-head self-attention",crossAttention:"encoder–decoder attention",feedForward:"position-wise feed-forward",output:"linear + softmax",addNorm:"residual + LayerNorm",modelWidth:"d_model",innerWidth:"d_ff",heads:"heads",keyWidth:"d_k = d_v",parameters:"parameters",steps:"train steps",dropout:"P_drop",smoothing:"ε_ls",stageLabel:"Inspect a stage",stageCopy:["Positional encodings are added to embeddings at the bottom of the encoder and decoder. Width remains d_model.","In self-attention, Q, K, and V come from the same previous-layer output. Every head has its own learned projections.","The decoder blocks future tokens by putting −∞ before softmax, together with output embeddings shifted by one position.","In encoder–decoder attention, Q comes from the decoder while K and V come from the encoder output.","Each position independently passes through two linear maps with ReLU between them: d_model → d_ff → d_model.","Every sub-layer is wrapped by a residual connection followed by LayerNorm—after every attention and FFN, not once per block."],stages:["embedding + position","self-attention","future mask","cross-attention","feed-forward","Add & Norm"]},attention:{kicker:"04 / EQUATION 1 · TEACHING SIMULATION",title:"Scaling is not decorative",intro:"As d_k grows, dot-product variance grows. Dividing by √d_k reduces the chance of pushing softmax into saturated regions with very small gradients.",objective:"Change d_k. The vectors really have the displayed width, and the divisor used is shown beside the result.",dimension:"Key dimension d_k",raw:"raw score",scaled:"scaled score",weight:"softmax weight",output:"Weighted output",divisor:"Actual divisor",inputNote:"Q, K, and V are deterministic teaching values, not model activations. This explains Equation 1; it does not reproduce a paper experiment."},position:{kicker:"05 / POSITION AND SHAPES",title:"Order enters from outside attention",intro:"With no recurrence or convolution, positional encodings are added to embeddings. Each dimension pair shares a frequency: sine at 2i and cosine at 2i+1.",position:"Position pos",pairNote:"d0/d1 are a pair, d2/d3 are a pair, and so on.",shapesTitle:"Shape trace",sequence:"Sequence length n",beforeProjection:"Before projection",perHead:"Inside each head",concatenated:"After concat",shapeNote:"Both base and big use d_k=d_v=64; head count and concatenated width are what change.",complexityTitle:"Compare asymptotic terms",complexityIntro:"These are the Table 1 terms with constants removed. They are relative operation counts, not execution time.",kernel:"Kernel width k",neighborhood:"Restricted window r",crossoverBefore:"At this length, the self-attention term is below the recurrent term.",crossoverAfter:"At n ≥ d_model, the quadratic term catches or exceeds recurrence."},training:{kicker:"06 / TRAINING",title:"The setup is part of the result",intro:"BLEU is not separable from data, checkpoint averaging, and decoding. Those details are part of the measurement instrument.",facts:[["WMT14 EN–DE","≈4.5M pairs · shared BPE ≈37K"],["WMT14 EN–FR","36M pairs · word-piece 32K"],["Batch","≈25K source + ≈25K target tokens"],["Hardware","1 machine · 8 × NVIDIA P100"],["Adam","β₁=.9 · β₂=.98 · ε=10⁻⁹"],["Decode","beam=4 · α=.6 · max=input+50"]],scheduleTitle:"Equation 3 / learning-rate schedule",scheduleIntro:"The rate rises linearly for 4000 steps, then falls with the inverse square root. d_model scales the entire curve.",step:"step",rate:"learning rate",warmup:"warmup=4000",checkpoint:"Base results average the last 5 checkpoints; big averages the last 20. Table 3 ablations use no checkpoint averaging. The EN–FR big model used P_drop=0.1 rather than 0.3."},results:{kicker:"07 / RESULTS",title:"The headline is clear. Its source needs precision.",intro:"Table 2 compares newstest2014 BLEU and estimated training FLOPs. Choose a task; every reported row is included.",german:"EN→DE",french:"EN→FR",model:"Model",type:"Type",bleu:"BLEU",cost:"Training cost",single:"single",ensemble:"ensemble",transformer:"Transformer",missing:"not reported",discrepancyTitle:"The source is internally inconsistent",discrepancy:"The abstract and Table 2 report 41.8 for EN→FR; §6.1 prose says 41.0. We show 41.8 as the table/abstract value and keep the conflict visible instead of resolving it silently.",ablationTitle:"Head count: more is not always better",ablationIntro:"Rows A vary h and d_k=d_v while holding approximate compute constant. Evaluation is on newstest2013 without checkpoint averaging.",heads:"heads",devBleu:"dev BLEU",devPpl:"dev PPL",finding:"One head is 0.9 BLEU below the best value. But 32 heads drops to 25.4; the evidence supports multi-head attention, not a rule that more is always better.",otherTitle:"What the other ablations say",parsingTitle:"Generalization beyond translation",parsing:"On WSJ Section 23, a 4-layer Transformer reached 91.3 F1 with WSJ-only training and 92.7 semi-supervised. It was strong, but did not beat the table’s best row: 93.3 for an RNN Grammar.",training:"Training",f1:"WSJ 23 F1"},boundaries:{kicker:"08 / CLAIM BOUNDARIES",title:"What did the paper actually establish?",lead:"It established the viability of sequence-transduction layers without recurrence or convolution. The experimental space remained narrow.",items:[["Paper-stated","Full self-attention costs O(n²d). The paper suggests restricted attention for long sequences and leaves its evaluation to future work."],["Paper-stated","Decoder generation remained autoregressive and sequential; the conclusion names reducing this sequentiality as a future goal."],["Editorial analysis","Appendix attention maps appear related to syntax and semantics, but selected examples are not causal tests or an interpretability metric."],["Editorial analysis","The evidence covers two translation tasks and constituency parsing. It does not establish superiority across every modality, length, or task."]],closing:"The paper changed the computational path, and its evidence supports that change. A broader claim needs experiments that were not present here."},glossary:{kicker:"09 / TERMINOLOGY",title:"Language choices",intro:"Arabic is used where it is precise and natural; established ML terms stay in English where that is the actual working vocabulary.",term:"Term",definition:"Definition",editorial:"Editorial note"},footer:{title:"Source",citation:"Vaswani et al., “Attention Is All You Need,” NIPS 2017, arXiv:1706.03762v7.",note:"All reported results and tables come from the local PDF. Interactive calculations are explicitly labeled as teaching simulations.",top:"Back to top"}}},I=e=>{if(e.length===0||e.some(a=>!Number.isFinite(a)))throw new Error("Softmax requires at least one finite value");const t=Math.max(...e),n=e.map(a=>Math.exp(a-t)),s=n.reduce((a,r)=>a+r,0);return n.map(a=>a/s)},N=(e,t,n,s=!0)=>{var E;const a=e.length,r=((E=n[0])==null?void 0:E.length)??0;if(a===0||t.length===0||t.length!==n.length||t.some(h=>h.length!==a)||n.some(h=>h.length!==r)||r===0)throw new Error("Incompatible attention dimensions");const d=t.map(h=>h.reduce((f,$,y)=>f+$*e[y],0)),b=s?Math.sqrt(a):1,w=d.map(h=>h/b),T=I(w),q=Array.from({length:r},(h,f)=>T.reduce(($,y,P)=>$+y*n[P][f],0));return{rawScores:d,scaledScores:w,weights:T,output:q,divisor:b}},O=e=>{if(!Number.isInteger(e)||e<2)throw new Error("dKey must be an integer of at least 2");const t=Array.from({length:e},(a,r)=>(r*7%11-5)/5),n=[t.map((a,r)=>a*.72+(r%3-1)*.08),t.map((a,r)=>r%2===0?-a*.42:a*.28),t.map((a,r)=>(r*5%13-6)/7)];return{query:t,keys:n,values:[[1,.15],[.12,1],[.58,.62]]}},R=(e,t,n)=>{if(!Number.isInteger(e)||e<0||!Number.isInteger(t)||t<0||t>=n||!Number.isInteger(n)||n<2)throw new Error("Invalid positional encoding coordinates");const s=Math.floor(t/2),a=e/Math.pow(1e4,2*s/n);return t%2===0?Math.sin(a):Math.cos(a)},v=(e,t=512,n=4e3)=>{if(e<1||t<1||n<1)throw new Error("Learning-rate inputs must be positive");return Math.pow(t,-.5)*Math.min(Math.pow(e,-.5),e*Math.pow(n,-1.5))},_=(e,t,n,s=32)=>{if([e,t,n,s].some(a=>!Number.isFinite(a)||a<=0))throw new Error("Complexity inputs must be positive");return{attention:e*e*t,recurrent:e*t*t,convolution:n*e*t*t,restricted:s*e*t}},m={title:"Attention Is All You Need",authors:["Ashish Vaswani","Noam Shazeer","Niki Parmar","Jakob Uszkoreit","Llion Jones","Aidan N. Gomez","Łukasz Kaiser","Illia Polosukhin"],venue:"NIPS 2017",arxiv:"1706.03762v7"},x=[{model:"ByteNet",kind:"single",de:23.75,fr:null,costDe:null,costFr:null,sharedCost:null},{model:"Deep-Att + PosUnk",kind:"single",de:null,fr:39.2,costDe:null,costFr:1e20,sharedCost:null},{model:"GNMT + RL",kind:"single",de:24.6,fr:39.92,costDe:23e18,costFr:14e19,sharedCost:null},{model:"ConvS2S",kind:"single",de:25.16,fr:40.46,costDe:96e17,costFr:15e19,sharedCost:null},{model:"MoE",kind:"single",de:26.03,fr:40.56,costDe:2e19,costFr:12e19,sharedCost:null},{model:"Deep-Att + PosUnk Ensemble",kind:"ensemble",de:null,fr:40.4,costDe:null,costFr:8e20,sharedCost:null},{model:"GNMT + RL Ensemble",kind:"ensemble",de:26.3,fr:41.16,costDe:18e19,costFr:11e20,sharedCost:null},{model:"ConvS2S Ensemble",kind:"ensemble",de:26.36,fr:41.29,costDe:77e18,costFr:12e20,sharedCost:null},{model:"Transformer (base model)",kind:"transformer",de:27.3,fr:38.1,costDe:null,costFr:null,sharedCost:33e17},{model:"Transformer (big)",kind:"transformer",de:28.4,fr:41.8,costDe:null,costFr:null,sharedCost:23e18}],K={base:{label:"Transformer (base)",layers:6,dModel:512,dFF:2048,heads:8,dKey:64,dValue:64,dropout:.1,labelSmoothing:.1,steps:1e5,parameters:65e6,devPpl:4.92,devBleu:25.8},big:{label:"Transformer (big)",layers:6,dModel:1024,dFF:4096,heads:16,dKey:64,dValue:64,dropout:.3,labelSmoothing:.1,steps:3e5,parameters:213e6,devPpl:4.33,devBleu:26.4}},B=[{heads:1,dKey:512,dValue:512,ppl:5.29,bleu:24.9},{heads:4,dKey:128,dValue:128,ppl:5,bleu:25.5},{heads:8,dKey:64,dValue:64,ppl:4.92,bleu:25.8},{heads:16,dKey:32,dValue:32,ppl:4.91,bleu:25.8},{heads:32,dKey:16,dValue:16,ppl:5.01,bleu:25.4}],C=[{group:"B",variable:"dₖ",comparison:"64 → 16",result:"BLEU 25.8 → 25.1",ar:"تقليل بُعد المفتاح أضعف الجودة؛ الورقة تقرأ هذا كإشارة إلى أن حساب التوافق مو مهمة سهلة.",en:"Reducing key dimension hurt quality; the paper reads this as evidence that compatibility is not easy to determine."},{group:"C",variable:"capacity",comparison:"d_model 512 → 1024",result:"BLEU 25.8 → 26.0",ar:"النموذج الأعرض تحسن، لكنه رفع عدد المعاملات من 65M إلى 168M في هذا الصف.",en:"The wider model improved, while parameter count rose from 65M to 168M in this row."},{group:"D",variable:"P_drop",comparison:"0.1 → 0.0",result:"BLEU 25.8 → 24.6",ar:"إزالة dropout أضرت بالنتيجة بوضوح تحت إعداد التطوير نفسه.",en:"Removing dropout clearly hurt the result under the same development setup."},{group:"E",variable:"position",comparison:"sinusoidal → learned",result:"BLEU 25.8 → 25.7",ar:"الـ positional embeddings المتعلّمة أعطت نتيجة شبه مطابقة للنسخة الجيبية.",en:"Learned positional embeddings produced a nearly identical result to the sinusoidal version."}],W=[{model:"Petrov et al. (2006)",training:"WSJ only, discriminative",f1:90.4},{model:"Transformer (4 layers)",training:"WSJ only, discriminative",f1:91.3},{model:"McClosky et al. (2006)",training:"semi-supervised",f1:92.1},{model:"Transformer (4 layers)",training:"semi-supervised",f1:92.7},{model:"Luong et al. (2015)",training:"multi-task",f1:93},{model:"Dyer et al. (2016)",training:"generative",f1:93.3}],U=[{name:"Self-attention",complexity:"O(n² · d)",sequential:"O(1)",path:"O(1)"},{name:"Recurrent",complexity:"O(n · d²)",sequential:"O(n)",path:"O(n)"},{name:"Convolutional",complexity:"O(k · n · d²)",sequential:"O(1)",path:"O(logₖ(n))"},{name:"Restricted self-attention",complexity:"O(r · n · d)",sequential:"O(1)",path:"O(n / r)"}],D=[{ar:"الانتباه الذاتي",en:"self-attention",definition:{ar:"عملية تأتي فيها Q وK وV من التسلسل نفسه، فتربط مواضعه ببعضها.",en:"Attention in which Q, K, and V come from the same sequence."},note:{ar:"نستخدم «الانتباه الذاتي» في الشرح، ونبقي self-attention داخل المخططات.",en:"The Arabic prose uses الانتباه الذاتي while diagrams retain self-attention."}},{ar:"رأس الانتباه",en:"attention head",definition:{ar:"مسار إسقاط وانتباه مستقل داخل multi-head attention.",en:"One learned projection-and-attention path inside multi-head attention."},note:{ar:"الرأس مو توزيع attention فقط؛ يشمل إسقاطات Q وK وV الخاصة به.",en:"A head is not only an attention distribution; it includes its Q, K, and V projections."}},{ar:"الترميز الموضعي",en:"positional encoding",abbreviation:"PE",definition:{ar:"إشارة تُجمع مع embedding عشان يدخل ترتيب المواضع إلى نموذج بلا recurrence أو convolution.",en:"A signal added to embeddings so a model without recurrence or convolution can use order."},note:{ar:"الورقة تختبر نسخة جيبية ثابتة وأخرى متعلّمة، والنتيجتان متقاربتان.",en:"The paper tests fixed sinusoidal and learned variants, with nearly identical results."}},{ar:"قناع المستقبل",en:"future mask",definition:{ar:"قناع يضع −∞ على الاتصالات غير المسموحة قبل softmax.",en:"A mask that places −∞ on disallowed connections before softmax."},note:{ar:"نسميه «قناع المستقبل» لأنه يصف وظيفة الورقة مباشرة.",en:"It preserves the decoder’s autoregressive property."}},{ar:"تعقيد كل طبقة",en:"per-layer complexity",definition:{ar:"صيغة asymptotic لعدد العمليات مع تغيّر n وd؛ مو latency مقاساً.",en:"An asymptotic operation count as n and d vary, not measured latency."},note:{ar:"المقارنة تستبعد ثوابت التنفيذ، حركة الذاكرة، وكفاءة الـ kernels.",en:"The comparison excludes implementation constants, memory movement, and kernel efficiency."}},{ar:"تنعيم التسميات",en:"label smoothing",definition:{ar:"regularization بقيمة ε_ls=0.1 في الإعداد الأساسي للورقة.",en:"Regularization used with ε_ls=0.1 in the paper’s base setup."},note:{ar:"الورقة تقول إنه يضر perplexity لكنه يحسن accuracy وBLEU.",en:"The paper reports worse perplexity but improved accuracy and BLEU."}}],i={locale:new URLSearchParams(location.search).get("lang")==="en"?"en":"ar",model:"base",architectureStage:0,dKey:64,position:5,sequenceLength:64,kernelSize:3,neighborhood:32,step:4e3,metric:"de",menuOpen:!1};let M=!1;const A=document.querySelector("#app");if(!A)throw new Error("Missing #app");const V=A,u=(e,t=2)=>new Intl.NumberFormat("en",{maximumFractionDigits:t}).format(e),k=e=>new Intl.NumberFormat("en",{notation:"compact",maximumFractionDigits:1}).format(e),j=e=>{if(e===null)return"—";const t=Math.floor(Math.log10(e)),n=e/10**t;return`${u(n,1)} × 10<sup>${t}</sup>`},l=e=>`<span class="citation" aria-label="Paper reference ${e}">${e}</span>`,g=(e,t)=>`<span class="evidence-tag evidence-tag--${t}">${e}</span>`,c=e=>`<svg aria-hidden="true" viewBox="0 0 24 24">${{sun:'<circle cx="12" cy="12" r="3.5"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/>',menu:'<path d="M4 7h16M4 12h16M4 17h16"/>',close:'<path d="m6 6 12 12M18 6 6 18"/>',arrow:'<path d="M5 12h14M13 6l6 6-6 6"/>',external:'<path d="M14 5h5v5M19 5l-8 8"/><path d="M18 13v5H6V6h5"/>'}[e]}</svg>`;function H(){return F[i.locale]}function p(){const e=H(),t=K[i.model];i.neighborhood=Math.min(i.neighborhood,i.sequenceLength);const n=O(i.dKey),s=N(n.query,n.keys,n.values),a=N(n.query,n.keys,n.values,!1),r=Array.from({length:8},(d,b)=>R(i.position,b,t.dModel)),o=_(i.sequenceLength,t.dModel,i.kernelSize,i.neighborhood);if(document.documentElement.lang=i.locale,document.documentElement.dir=i.locale==="ar"?"rtl":"ltr",document.title=i.locale==="ar"?`${m.title} — قراءة تفاعلية`:`${m.title} — interactive reading`,V.innerHTML=`
    <a class="skip-link" href="#main">${e.ui.skip}</a>
    <div class="reading-progress" aria-hidden="true"><i id="reading-progress"></i></div>

    <header class="site-header">
      <a class="wordmark" href="#top" aria-label="${m.title}">
        <span>VOIDWEST</span><i></i><small>PAPER / 001</small>
      </a>
      <div class="header-actions">
        <a class="source-link" href="./paper/1706.03762v7.pdf">
          ${e.ui.source}${c("external")}
        </a>
        <button class="text-button" data-action="language">${e.ui.language}</button>
        <button class="icon-button" data-action="theme" aria-label="${e.ui.theme}">
          ${c("sun")}
        </button>
        <button class="icon-button menu-button" data-action="menu" aria-label="${i.menuOpen?e.ui.close:e.ui.menu}" aria-expanded="${i.menuOpen}">
          ${c(i.menuOpen?"close":"menu")}
        </button>
      </div>
    </header>

    ${Q(e)}

    <main id="main">
      ${G(e)}
      <div class="article-shell">
        ${z(e)}
        <article>
          ${J(e)}
          ${Y(e)}
          ${X(e,t)}
          ${Z(e,s,a)}
          ${ee(e,t,r,o)}
          ${ae(e,t)}
          ${ne(e)}
          ${ie(e)}
          ${se(e)}
          ${re(e)}
        </article>
      </div>
    </main>
  `,oe(),S(),!M){M=!0;const d=location.hash?document.querySelector(location.hash):null;d&&d.scrollIntoView()}}function Q(e){const t=L(e);return`
    <div class="mobile-nav ${i.menuOpen?"is-open":""}" ${i.menuOpen?"":"hidden"}>
      <nav aria-label="${e.ui.menu}">
        <span class="micro-label">${e.ui.menu}</span>
        ${t}
      </nav>
    </div>
  `}function L(e){return[["overview","01",e.nav.overview],["comparison","02",e.nav.comparison],["architecture","03",e.nav.architecture],["attention","04",e.nav.attention],["position","05",e.nav.position],["training","06",e.nav.training],["results","07",e.nav.results],["boundaries","08",e.nav.boundaries],["glossary","09",e.nav.glossary]].map(([n,s,a])=>`<a href="#${n}" data-nav-link><small>${s}</small><span>${a}</span></a>`).join("")}function z(e){return`
    <aside class="side-nav">
      <span class="micro-label">${e.ui.menu}</span>
      <nav aria-label="${e.ui.menu}">${L(e)}</nav>
    </aside>
  `}function G(e){return`
    <section class="hero" id="top">
      <div class="hero-main">
        <span class="eyebrow">${e.hero.eyebrow}</span>
        <h1>${m.title}</h1>
        <p class="hero-dek">${e.hero.dek}</p>
        <p class="hero-contribution">${e.hero.contribution}</p>
        <p class="authors" dir="ltr">${m.authors.join(" · ")}</p>
        <div class="hero-meta">
          <span>${m.venue}</span>
          <span>${m.arxiv}</span>
          <span>${e.hero.reading}</span>
          <span>${e.hero.prerequisites}</span>
        </div>
        <p class="disclosure">${e.hero.disclosure}</p>
        <div class="hero-actions">
          <a class="button button--solid" href="./paper/1706.03762v7.pdf">${e.hero.openPaper}${c("external")}</a>
          <a class="button button--quiet" href="#overview">${e.hero.begin}${c("arrow")}</a>
        </div>
      </div>
      <aside class="hero-results" aria-label="${e.ui.reported}">
        <span class="micro-label">${e.ui.reported} · Table 2</span>
        <div>
          <strong>28.4</strong>
          <span>BLEU</span>
          <p>${e.hero.resultLabel}</p>
          <small>${e.hero.resultNote}</small>
        </div>
        <div>
          <strong>41.8</strong>
          <span>BLEU</span>
          <p>${e.hero.secondLabel}</p>
          <small>${e.hero.secondNote}</small>
        </div>
      </aside>
    </section>
  `}function J(e){return`
    <section class="article-section" id="overview">
      <header class="section-header">
        <span>${e.overview.kicker}</span>
        <h2>${e.overview.title}</h2>
      </header>
      <p class="lead">${e.overview.lead}</p>
      <p>${e.overview.body} ${l("Abstract · §1 · §6")}</p>
      <div class="claim-grid">
        <div>
          ${g(e.ui.reported,"paper")}
          <h3>${e.overview.claimTitle}</h3>
          <p>${e.overview.claim}</p>
        </div>
        <div>
          ${g(e.ui.editorial,"editorial")}
          <h3>${e.overview.boundaryTitle}</h3>
          <p>${e.overview.boundary}</p>
        </div>
      </div>
      <blockquote>
        <span>${e.overview.mechanismTitle}</span>
        <p>${e.overview.mechanism}</p>
      </blockquote>
    </section>
  `}function Y(e){return`
    <section class="article-section" id="comparison">
      <header class="section-header">
        <span>${e.comparison.kicker}</span>
        <h2>${e.comparison.title}</h2>
      </header>
      <p class="lead">${e.comparison.intro}</p>
      <div class="table-scroll">
        <table class="data-table comparison-table">
          <thead><tr>
            <th>${e.comparison.layer}</th>
            <th>${e.comparison.cost}</th>
            <th>${e.comparison.sequential}</th>
            <th>${e.comparison.path}</th>
          </tr></thead>
          <tbody>
            ${U.map((t,n)=>`<tr class="${n===0?"is-highlighted":""}">
                  <th>${t.name}</th><td dir="ltr">${t.complexity}</td>
                  <td dir="ltr">${t.sequential}</td><td dir="ltr">${t.path}</td>
                </tr>`).join("")}
          </tbody>
        </table>
      </div>
      <p class="source-note">${e.comparison.note} ${l("§4 · Table 1")}</p>
    </section>
  `}function X(e,t){const n=(s,a="")=>`<div class="architecture-node ${a}"><span>${s}</span></div>`;return`
    <section class="article-section" id="architecture">
      <header class="section-header">
        <span>${e.architecture.kicker}</span>
        <h2>${e.architecture.title}</h2>
      </header>
      <p class="lead">${e.architecture.intro} ${l("§3 · Table 3")}</p>
      <div class="segmented" role="group" aria-label="Model configuration">
        <button data-model="base" class="${i.model==="base"?"is-active":""}">${e.architecture.base}</button>
        <button data-model="big" class="${i.model==="big"?"is-active":""}">${e.architecture.big}</button>
      </div>
      <div class="model-specs" aria-live="polite">
        ${[[e.architecture.modelWidth,t.dModel],[e.architecture.innerWidth,t.dFF],[e.architecture.heads,t.heads],[e.architecture.keyWidth,t.dKey],[e.architecture.parameters,k(t.parameters)],[e.architecture.steps,k(t.steps)],[e.architecture.dropout,t.dropout],[e.architecture.smoothing,t.labelSmoothing]].map(([s,a])=>`<div><span>${s}</span><strong dir="ltr">${a}</strong></div>`).join("")}
      </div>
      <div class="architecture-map" dir="ltr">
        <div class="architecture-stack">
          <h3>${e.architecture.encoder}</h3>
          ${n(e.architecture.input,"architecture-node--input")}
          <div class="architecture-block">
            ${n(e.architecture.selfAttention)}
            ${n(e.architecture.addNorm,"architecture-node--norm")}
            ${n(`${e.architecture.feedForward} · ${t.dModel} → ${t.dFF} → ${t.dModel}`)}
            ${n(e.architecture.addNorm,"architecture-node--norm")}
          </div>
          <small>× ${t.layers}</small>
        </div>
        <div class="architecture-bridge" aria-hidden="true"><i></i><span>K, V</span></div>
        <div class="architecture-stack">
          <h3>${e.architecture.decoder}</h3>
          ${n(e.architecture.input,"architecture-node--input")}
          <div class="architecture-block">
            ${n(e.architecture.masked)}
            ${n(e.architecture.addNorm,"architecture-node--norm")}
            ${n(e.architecture.crossAttention)}
            ${n(e.architecture.addNorm,"architecture-node--norm")}
            ${n(`${e.architecture.feedForward} · ${t.dModel} → ${t.dFF} → ${t.dModel}`)}
            ${n(e.architecture.addNorm,"architecture-node--norm")}
          </div>
          <small>× ${t.layers}</small>
          ${n(e.architecture.output,"architecture-node--input")}
        </div>
      </div>
      <div class="stage-inspector">
        <span class="micro-label">${e.architecture.stageLabel}</span>
        <div class="stage-tabs" role="tablist">
          ${e.architecture.stages.map((s,a)=>`<button role="tab" aria-selected="${i.architectureStage===a}" data-stage="${a}" class="${i.architectureStage===a?"is-active":""}">${String(a+1).padStart(2,"0")} / ${s}</button>`).join("")}
        </div>
        <p class="stage-copy" aria-live="polite">${e.architecture.stageCopy[i.architectureStage]} ${l("§3.1–3.5 · Fig. 1")}</p>
      </div>
    </section>
  `}function Z(e,t,n){return`
    <section class="article-section interactive-section" id="attention">
      <header class="section-header">
        <span>${e.attention.kicker}</span>
        <h2>${e.attention.title}</h2>
      </header>
      <p class="lead">${e.attention.intro}</p>
      <div class="equation" dir="ltr" role="img" aria-label="Attention of Q K V equals softmax of Q K transpose divided by square root of d k, multiplied by V">
        <span>Attention(Q, K, V)</span>
        <i>=</i>
        <span>softmax( QK<sup>T</sup> / √d<sub>k</sub> )V</span>
      </div>
      <div class="lab">
        <div class="lab-control">
          ${g(e.ui.simulation,"simulation")}
          <p>${e.attention.objective}</p>
          <label for="d-key"><span>${e.attention.dimension}</span><output>${i.dKey}</output></label>
          <input id="d-key" data-input="dKey" type="range" min="16" max="128" step="16" value="${i.dKey}">
          <div class="lab-stat">
            <span>${e.attention.divisor}</span>
            <strong dir="ltr">√${i.dKey} = ${u(t.divisor,4)}</strong>
          </div>
          <div class="lab-stat">
            <span>${e.attention.output}</span>
            <strong dir="ltr">[${t.output.map(s=>u(s,4)).join(", ")}]</strong>
          </div>
          <button class="reset-button" data-reset="attention">${e.ui.reset}</button>
        </div>
        <div class="attention-readout" aria-live="polite">
          <div class="attention-head">
            <span>Key</span><span>${e.attention.raw}</span><span>${e.attention.scaled}</span><span>${e.attention.weight}</span>
          </div>
          ${t.weights.map((s,a)=>`
              <div class="attention-row">
                <b>K${a+1}</b>
                <span dir="ltr">${u(t.rawScores[a],4)}</span>
                <span dir="ltr">${u(t.scaledScores[a],4)}</span>
                <div class="weight-cell">
                  <i style="--weight:${s}"></i>
                  <strong dir="ltr">${u(s,4)}</strong>
                </div>
              </div>`).join("")}
          <p class="comparison-readout" dir="ltr">
            max softmax weight · scaled ${u(Math.max(...t.weights),4)}
            / unscaled ${u(Math.max(...n.weights),4)}
          </p>
        </div>
      </div>
      <p class="source-note">${e.attention.inputNote} ${l("§3.2.1 · Eq. 1 · footnote 4")}</p>
    </section>
  `}function ee(e,t,n,s){const a=Math.max(...Object.values(s)),r={attention:"self-attention",recurrent:"recurrent",convolution:`convolution · k=${i.kernelSize}`,restricted:`restricted · r=${i.neighborhood}`};return`
    <section class="article-section" id="position">
      <header class="section-header">
        <span>${e.position.kicker}</span>
        <h2>${e.position.title}</h2>
      </header>
      <p class="lead">${e.position.intro} ${l("§3.5")}</p>
      <div class="split-lab">
        <div class="position-panel">
          ${g(e.ui.simulation,"simulation")}
          <label for="position-input"><span>${e.position.position}</span><output>${i.position}</output></label>
          <input id="position-input" data-input="position" type="range" min="0" max="40" value="${i.position}">
          <div class="position-values" dir="ltr">
            ${n.map((o,d)=>`<div>
                  <span>d${d}</span>
                  <i><b style="--value:${(o+1)/2}"></b></i>
                  <strong>${o.toFixed(3)}</strong>
                </div>`).join("")}
          </div>
          <div class="equation equation--small" dir="ltr">
            PE(pos,2i)=sin(pos/10000<sup>2i/d_model</sup>)<br>
            PE(pos,2i+1)=cos(pos/10000<sup>2i/d_model</sup>)
          </div>
          <p class="micro-copy">${e.position.pairNote}</p>
          <button class="reset-button" data-reset="position">${e.ui.reset}</button>
        </div>
        <div class="shape-panel">
          <h3>${e.position.shapesTitle}</h3>
          <label for="sequence-input"><span>${e.position.sequence}</span><output>${i.sequenceLength}</output></label>
          <input id="sequence-input" data-input="sequenceLength" type="range" min="16" max="1024" step="16" value="${i.sequenceLength}">
          <div class="shape-trace" dir="ltr">
            <div><span>${e.position.beforeProjection}</span><strong>[n, ${t.dModel}]</strong></div>
            <i>${c("arrow")}</i>
            <div><span>${e.position.perHead}</span><strong>${t.heads} × [n, ${t.dKey}]</strong></div>
            <i>${c("arrow")}</i>
            <div><span>${e.position.concatenated}</span><strong>[n, ${t.heads*t.dValue}]</strong></div>
          </div>
          <p class="micro-copy">${e.position.shapeNote}</p>
        </div>
      </div>

      <div class="complexity-lab">
        <div class="complexity-copy">
          <h3>${e.position.complexityTitle}</h3>
          <p>${e.position.complexityIntro}</p>
          <div class="compact-controls">
            <label for="kernel-input"><span>${e.position.kernel}</span><output>${i.kernelSize}</output></label>
            <input id="kernel-input" data-input="kernelSize" type="range" min="1" max="9" step="2" value="${i.kernelSize}">
            <label for="neighborhood-input"><span>${e.position.neighborhood}</span><output>${i.neighborhood}</output></label>
            <input id="neighborhood-input" data-input="neighborhood" type="range" min="8" max="${Math.min(128,i.sequenceLength)}" step="8" value="${i.neighborhood}">
          </div>
        </div>
        <div class="cost-bars" aria-live="polite">
          ${Object.entries(s).map(([o,d])=>`<div>
                <span>${r[o]}</span>
                <i><b style="width:${Math.max(1.5,d/a*100)}%"></b></i>
                <strong dir="ltr">${k(d)}</strong>
              </div>`).join("")}
          <p>${i.sequenceLength<t.dModel?e.position.crossoverBefore:e.position.crossoverAfter} ${l("§4 · Table 1")}</p>
        </div>
      </div>
    </section>
  `}function te(e){const t=v(4e3,e);return Array.from({length:81},(n,s)=>{const a=Math.max(1,s*250),r=a/2e4*100,o=92-v(a,e)/t*78;return`${r.toFixed(2)},${o.toFixed(2)}`}).join(" ")}function ae(e,t){const n=v(i.step,t.dModel),s=v(4e3,t.dModel),a=i.step/2e4*100,r=92-n/s*78;return`
    <section class="article-section" id="training">
      <header class="section-header">
        <span>${e.training.kicker}</span>
        <h2>${e.training.title}</h2>
      </header>
      <p class="lead">${e.training.intro}</p>
      <dl class="fact-list">
        ${e.training.facts.map(([o,d])=>`<div><dt>${o}</dt><dd dir="ltr">${d}</dd></div>`).join("")}
      </dl>
      <div class="schedule-lab">
        <div>
          ${g(e.ui.simulation,"simulation")}
          <h3>${e.training.scheduleTitle}</h3>
          <p>${e.training.scheduleIntro}</p>
          <label for="step-input"><span>${e.training.step}</span><output>${u(i.step,0)}</output></label>
          <input id="step-input" data-input="step" type="range" min="1" max="20000" step="100" value="${i.step}">
          <div class="lab-stat"><span>${e.training.rate}</span><strong dir="ltr">${n.toExponential(6)}</strong></div>
          <div class="lab-stat"><span>d_model</span><strong dir="ltr">${t.dModel}</strong></div>
        </div>
        <div class="schedule-chart" dir="ltr" aria-label="Learning rate curve">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img">
            <path class="chart-grid" d="M0 92H100M0 53H100M0 14H100M20 5V95"/>
            <polyline points="${te(t.dModel)}"/>
            <circle cx="${a}" cy="${r}" r="2.2"/>
          </svg>
          <div><span>1</span><span>4K warmup</span><span>20K step</span></div>
        </div>
      </div>
      <p class="source-note">${e.training.checkpoint} ${l("§5 · §6.1 · Eq. 3")}</p>
    </section>
  `}function ne(e){const t=i.metric,n=Math.max(...x.map(a=>a[t]).filter(a=>a!==null)),s=x.filter(a=>a[t]!==null);return`
    <section class="article-section" id="results">
      <header class="section-header">
        <span>${e.results.kicker}</span>
        <h2>${e.results.title}</h2>
      </header>
      <p class="lead">${e.results.intro}</p>
      <div class="results-toolbar">
        ${g(e.ui.reported,"paper")}
        <div class="segmented" role="group" aria-label="Translation task">
          <button data-metric="de" class="${i.metric==="de"?"is-active":""}">${e.results.german}</button>
          <button data-metric="fr" class="${i.metric==="fr"?"is-active":""}">${e.results.french}</button>
        </div>
      </div>
      <div class="table-scroll">
        <table class="data-table result-table">
          <thead><tr>
            <th>${e.results.model}</th><th>${e.results.type}</th>
            <th>${e.results.bleu} <small>↑</small></th><th>${e.results.cost} · FLOPs</th>
          </tr></thead>
          <tbody>
            ${s.map(a=>{const r=a[t],o=a.sharedCost??(t==="de"?a.costDe:a.costFr);return`<tr class="${a.kind==="transformer"?"is-highlighted":""}">
                  <th>${a.model}</th>
                  <td>${e.results[a.kind]}</td>
                  <td><strong>${r}</strong>${r===n?'<span class="best-mark">best</span>':""}</td>
                  <td dir="ltr">${o===null?e.results.missing:j(o)}</td>
                </tr>`}).join("")}
          </tbody>
        </table>
      </div>
      <p class="source-note">${l("Table 2 · newstest2014")}</p>
      <aside class="source-conflict">
        <span>41.0 / 41.8</span>
        <div><h3>${e.results.discrepancyTitle}</h3><p>${e.results.discrepancy} ${l("Abstract · §6.1 · Table 2")}</p></div>
      </aside>

      <div class="subsection">
        <h3>${e.results.ablationTitle}</h3>
        <p>${e.results.ablationIntro}</p>
        <div class="ablation-chart" dir="ltr">
          ${B.map(a=>`<div class="${a.heads===8?"is-base":""}">
                <span>${a.bleu}</span>
                <i><b style="height:${(a.bleu-24)/2*100}%"></b></i>
                <strong>${a.heads}</strong>
                <small>h · d<sub>k</sub>=${a.dKey}<br>PPL ${a.ppl.toFixed(2)}</small>
              </div>`).join("")}
        </div>
        <p class="source-note">${e.results.finding} ${l("§6.2 · Table 3 rows A")}</p>
      </div>

      <div class="subsection">
        <h3>${e.results.otherTitle}</h3>
        <div class="variation-list">
          ${C.map(a=>`<div>
                <span>ROW ${a.group}</span>
                <h4 dir="ltr">${a.variable} · ${a.comparison}</h4>
                <strong dir="ltr">${a.result}</strong>
                <p>${i.locale==="ar"?a.ar:a.en}</p>
              </div>`).join("")}
        </div>
        <p class="source-note">${l("§6.2 · Table 3 rows B–E")}</p>
      </div>

      <div class="subsection">
        <h3>${e.results.parsingTitle}</h3>
        <p>${e.results.parsing}</p>
        <div class="table-scroll">
          <table class="data-table">
            <thead><tr><th>${e.results.model}</th><th>${e.results.training}</th><th>${e.results.f1} ↑</th></tr></thead>
            <tbody>
              ${W.map(a=>`<tr class="${a.model.startsWith("Transformer")?"is-highlighted":""}">
                    <th>${a.model}</th><td dir="ltr">${a.training}</td><td>${a.f1}</td>
                  </tr>`).join("")}
            </tbody>
          </table>
        </div>
        <p class="source-note">${l("§6.3 · Table 4")}</p>
      </div>
    </section>
  `}function ie(e){return`
    <section class="article-section" id="boundaries">
      <header class="section-header">
        <span>${e.boundaries.kicker}</span>
        <h2>${e.boundaries.title}</h2>
      </header>
      <p class="lead">${e.boundaries.lead}</p>
      <div class="boundary-list">
        ${e.boundaries.items.map(([t,n],s)=>`<div>
              ${g(t,s<2?"paper":"editorial")}
              <p>${n}</p>
            </div>`).join("")}
      </div>
      <p class="closing-line">${e.boundaries.closing} ${l("§4 · §7 · Appendix")}</p>
    </section>
  `}function se(e){return`
    <section class="article-section" id="glossary">
      <header class="section-header">
        <span>${e.glossary.kicker}</span>
        <h2>${e.glossary.title}</h2>
      </header>
      <p class="lead">${e.glossary.intro}</p>
      <div class="glossary-list">
        ${D.map((t,n)=>`<details ${n===0?"open":""}>
              <summary>
                <span><b>${t.ar}</b><small dir="ltr">${t.en}${t.abbreviation?` · ${t.abbreviation}`:""}</small></span>
                <i aria-hidden="true">+</i>
              </summary>
              <div>
                <p><strong>${e.glossary.definition}</strong>${t.definition[i.locale]}</p>
                <p><strong>${e.glossary.editorial}</strong>${t.note[i.locale]}</p>
              </div>
            </details>`).join("")}
      </div>
    </section>
  `}function re(e){return`
    <footer class="article-footer">
      <div>
        <span class="micro-label">${e.footer.title}</span>
        <p dir="ltr">${e.footer.citation}</p>
        <small>${e.footer.note}</small>
      </div>
      <div class="footer-links">
        <a href="./paper/1706.03762v7.pdf">Local PDF ${c("external")}</a>
        <a href="https://arxiv.org/abs/1706.03762">arXiv ${c("external")}</a>
        <a href="#top">${e.footer.top} ${c("arrow")}</a>
      </div>
    </footer>
  `}function oe(){document.querySelectorAll("[data-action]").forEach(t=>{t.addEventListener("click",()=>{const n=t.dataset.action;if(n==="language"){i.locale=i.locale==="ar"?"en":"ar";const s=new URL(location.href);s.searchParams.set("lang",i.locale),history.replaceState(null,"",s),p()}n==="theme"&&le(),n==="menu"&&(i.menuOpen=!i.menuOpen,p())})}),document.querySelectorAll("[data-nav-link]").forEach(t=>{t.addEventListener("click",()=>{i.menuOpen&&(i.menuOpen=!1,p())})}),document.querySelectorAll("[data-model]").forEach(t=>{t.addEventListener("click",()=>{i.model=t.dataset.model,p()})}),document.querySelectorAll("[data-stage]").forEach(t=>{t.addEventListener("click",()=>{var n;i.architectureStage=Number(t.dataset.stage),p(),(n=document.querySelector(`[data-stage="${i.architectureStage}"]`))==null||n.focus()})}),document.querySelectorAll("[data-metric]").forEach(t=>{t.addEventListener("click",()=>{i.metric=t.dataset.metric,p()})});const e={dKey:"dKey",position:"position",sequenceLength:"sequenceLength",kernelSize:"kernelSize",neighborhood:"neighborhood",step:"step"};document.querySelectorAll("[data-input]").forEach(t=>{t.addEventListener("change",()=>{var s;const n=e[t.dataset.input??""];n&&(i[n]=Number(t.value),p(),(s=document.querySelector(`#${t.id}`))==null||s.focus())})}),document.querySelectorAll("[data-reset]").forEach(t=>{t.addEventListener("click",()=>{t.dataset.reset==="attention"&&(i.dKey=64),t.dataset.reset==="position"&&(i.position=5),p()})})}function le(){var n;const e=document.documentElement.dataset.theme==="light"?"dark":"light";document.documentElement.dataset.theme=e,localStorage.setItem("paper-theme",e);const t=new URL(location.href);t.searchParams.delete("theme"),history.replaceState(null,"",t),(n=document.querySelector('meta[name="theme-color"]'))==null||n.setAttribute("content",e==="light"?"#f5f1e8":"#11110f")}function S(){var s;const e=document.documentElement,t=e.scrollHeight-e.clientHeight,n=t>0?e.scrollTop/t*100:0;(s=document.querySelector("#reading-progress"))==null||s.style.setProperty("width",`${n}%`)}window.addEventListener("scroll",S,{passive:!0});p();
