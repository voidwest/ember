# Ember demo outline

A five-minute live demo of the experiment console. It is built around one
contrast that was measured on a real model, not invented: the same kind of
change made in two different places has very different effects.

Model used to check this: Llama-3.2-1B-Instruct-Q8_0, prompt
"The capital of France is", change applied at the last prompt token.

| Change | What happens to the words |
|---|---|
| Silence layer 6's output (scale x0.0) | "Paris. The Eiffel Tower is located in Paris…" becomes "covered in a thick layer of fog, and the streets are eerily quiet…" |
| Zero the MLP at layer 12 | Unchanged. The internals move (relative L2 0.43) but the output does not. |

## Before you start

- Build the release binary: `cargo build --release --bin ember`, then
  `./target/release/ember gui`. A debug build makes the first run take more than
  a minute.
- Run the whole flow once on the presenting machine, with the same model and the
  same screen. Loading the model is the slow part; do it before people arrive
  (Models -> select -> Load, or just run once).
- Presenting on a projector or from a distance: Cmd+K -> Presentation mode
  (larger text, sidebar and inspector hidden). Cmd+K again to leave it.
- Keep **See a sample result** in mind as the fallback. It opens a finished,
  clearly labelled illustrative comparison and needs no model.

## The talk track

**1. The question (30s) -- Home**
"Language models are opaque. Ember lets you reach inside one, change a single
thing, and see what that does." Point at *How Ember works*: Ask, Change one
thing, Compare.

**2. Ask (30s) -- Try an example**
Click **Try an example**. It loads *Silence an early layer* into the workspace:
the setup is on the left, the results will land on the right. Point at the prompt
("The capital of France is") and the model. The baseline is the untouched model.

**3. Change one thing (45s) -- Setup**
Read the *Change* section aloud: scale layer 6's output to zero at the last
prompt token. The sentence under it says what the change does. Everything you
would want to tweak is on this one pane; nothing is behind a step.

**4. Run and compare (60s) -- Results**
Press **Run experiment** (Cmd+Enter). Narrate the progress steps: load, run
baseline and intervention, compare. When it lands:
- Read the verdict: *Output changed, first differs at step 1*.
- Baseline vs Intervention side by side: the model stopped answering the
  question and started describing fog.
- Hover a term (*Peak divergence*, *First internal divergence*) for the
  plain-language definition.

**5. Where inside the model (45s) -- Layers**
Open the Layers tab. The dashed vertical line is the layer you touched; nothing
before it moves, and everything after it does.

**6. The contrast, and the point of the tool (75s) -- Pin and compare**
Click **Pin as reference**. Now change one thing in the setup: *Change* to
*Remove information*, *Where* to *After MLP block*, layer 12. Note that the
results immediately say **Settings changed since this result**. Press **Run
again**. Because the model is already loaded it takes about a second. The
*Compared with pinned reference* panel shows both runs side by side (about
-63% peak divergence), the words are unchanged this time, and the Layers chart
draws the pinned run as a dashed line behind the new one. "Same idea, a
different place: the model absorbs it. Where you intervene is the whole story."

**6b. The whole curve (45s, optional) -- Sweep**
Set *Change* back to *Change strength*, strength 0.0, *Where* to *Layer output*,
then click **Sweep all 16 layers**. About twenty seconds later the Sweep tab
shows how much each layer's change moved the model; the words changed at every
layer here. Click **Open** on a row to read that run.

**7. Take it with you (15s) -- Copy summary**
Click **Copy summary** and paste it anywhere: model, prompt, change, verdict, both
outputs and the per-layer table, as Markdown.

## If something goes wrong

- Model will not load or the run is slow: open **See a sample result** from Home
  and walk steps 4 to 6 on the sample. It is labelled illustrative.
- Text too small: Cmd+K -> Presentation mode.
- You want a cleaner start: Runs -> delete old runs; the sample and examples do
  not depend on history.

## Things to be honest about

- The two outcomes above were measured once, on a 16-layer model with a 24-token
  cap. Other models and prompts will differ; that difference is part of the point.
- The sample result is illustrative, not a measurement, and says so on the page.
- Keyboard focus rings and full-window interaction were checked in the code and
  in offscreen renders, not by hand on a live window; click through once first.
