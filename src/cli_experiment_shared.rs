//! Shared-prefix execution for experiments that run together.
//!
//! One *base* run computes the prompt prefill once. While it runs:
//!
//! * *co-baselines* (non-intervening specs with the same inputs, settings
//!   and generated-step sites) observe its entire generation and get their
//!   own bundles without a generation of their own;
//! * an *observer* instance of each *variant* sees the prefill blocks before
//!   that variant's boundary (`ember::v05::prefix`), recording the variant's
//!   early captures;
//! * the residual stream entering each boundary block and the KV cache
//!   after prefill are recorded.
//!
//! Each variant then starts its prefill at its boundary from the recorded
//! state, with the observer's captures handed over, and runs its decode as
//! usual. Anything that cannot be shared safely runs in full, and every
//! bundle's `runtime.json` records the path it took. Bundles are
//! bit-identical to standalone runs; only `runtime.json` differs.

use crate::cli_experiment::{
    activate_spec, ensure_same_session, finish_bundle, new_input_experiment, pool_threads,
    run_input, ActiveSpec, PreparedRun, RunOutcome, RunTarget, RunTiming,
};
use crate::cli_generation::PrefixRole;
use anyhow::Context;
use ember::artifact::ActivationStage;
use ember::cancel::CancelToken;
use ember::experiments::{
    ExecutionContext, ExecutionPhase, Experiment, ExperimentError, GenerationContext, LayerContext,
    ModelContext, TensorAccess,
};
use ember::kv_cache::KVCache;
use ember::tensor::CpuTensor;
use ember::v05::prefix::{
    co_baseline_mismatch, resume_decision, PrefixInputRecord, PrefixPath, PrefixReuseRecord,
    ReuseDecision,
};
use ember::v05::runner::{InputResult, PrefixObservations, V05Experiment};
use ember::v05::spec::ExperimentSpecV1;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

type Shared<T> = Arc<Mutex<T>>;

fn lock<T>(value: &Shared<T>) -> std::sync::MutexGuard<'_, T> {
    value.lock().expect("shared experiment lock")
}

/// What the base pass records beyond the experiments' own results.
#[derive(Default)]
struct PassRecord {
    prompt: Vec<u32>,
    hidden: BTreeMap<usize, CpuTensor>,
    co_failures: Vec<Option<String>>,
    observer_failures: Vec<Option<String>>,
}

/// The composite experiment of a base pass (see the module docs).
pub(crate) struct SharedPass {
    base: Shared<V05Experiment>,
    co: Vec<Shared<V05Experiment>>,
    /// Variant observers and their boundaries.
    observers: Vec<(Shared<V05Experiment>, usize)>,
    /// Boundaries whose entering hidden state is recorded.
    record_layers: BTreeSet<usize>,
    record: Shared<PassRecord>,
}

impl SharedPass {
    fn co_failed(&self, index: usize) -> bool {
        lock(&self.record).co_failures[index].is_some()
    }

    fn observer_failed(&self, index: usize) -> bool {
        lock(&self.record).observer_failures[index].is_some()
    }

    fn fail_co(&self, index: usize, error: ExperimentError) {
        lock(&self.record).co_failures[index] = Some(error.message().to_string());
    }

    fn fail_observer(&self, index: usize, reason: String) {
        lock(&self.record).observer_failures[index] = Some(reason);
    }

    /// Fire a per-layer hook: the base first (its result is authoritative),
    /// then every live co-baseline, then -- during prefill, before their
    /// boundaries -- every live observer. Only the base may mutate, and it
    /// does not before any boundary; the others never intervene where they
    /// are fired, so their order is immaterial.
    fn fire_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        tensor: &mut TensorAccess<'_>,
        hook: fn(
            &mut V05Experiment,
            &LayerContext<'_>,
            &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError>,
    ) -> Result<(), ExperimentError> {
        hook(&mut lock(&self.base), ctx, tensor)?;
        for index in 0..self.co.len() {
            if !self.co_failed(index)
                && let Err(error) = hook(&mut lock(&self.co[index]), ctx, tensor)
            {
                self.fail_co(index, error);
            }
        }
        if ctx.execution.phase != ExecutionPhase::Prefill {
            return Ok(());
        }
        let layer = ctx.layer_index;
        for index in 0..self.observers.len() {
            let (observer, boundary) = &self.observers[index];
            if layer >= *boundary || self.observer_failed(index) {
                continue;
            }
            let mut observer = lock(observer);
            let outcome = if observer.intervenes_in_prefill_layer(layer) {
                Err(format!(
                    "the variant intervenes in block {layer}, before its boundary {boundary}"
                ))
            } else {
                hook(&mut observer, ctx, tensor).map_err(|error| error.message().to_string())
            };
            drop(observer);
            if let Err(reason) = outcome {
                self.fail_observer(index, reason);
            }
        }
        Ok(())
    }

    fn fire_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        tensor: &mut TensorAccess<'_>,
        hook: fn(
            &mut V05Experiment,
            &ExecutionContext<'_>,
            &mut TensorAccess<'_>,
        ) -> Result<(), ExperimentError>,
    ) -> Result<(), ExperimentError> {
        hook(&mut lock(&self.base), ctx, tensor)?;
        for index in 0..self.co.len() {
            if !self.co_failed(index)
                && let Err(error) = hook(&mut lock(&self.co[index]), ctx, tensor)
            {
                self.fail_co(index, error);
            }
        }
        Ok(())
    }
}

impl Experiment for SharedPass {
    fn name(&self) -> &'static str {
        "v05-experiment"
    }

    fn intervenes(&self) -> bool {
        lock(&self.base).intervenes()
    }

    fn uses_activation_site(
        &self,
        stage: ActivationStage,
        layer: Option<usize>,
        phase: ExecutionPhase,
    ) -> bool {
        // Co-baselines hook the base's generated-step sites exactly, so in
        // decode (the only phase whose route depends on this) the union is
        // the base's own answer.
        lock(&self.base).uses_activation_site(stage, layer, phase)
            || self
                .co
                .iter()
                .any(|co| lock(co).uses_activation_site(stage, layer, phase))
            || (phase == ExecutionPhase::Prefill
                && self.observers.iter().any(|(observer, boundary)| {
                    layer.is_some_and(|layer| layer < *boundary)
                        && lock(observer).uses_activation_site(stage, layer, phase)
                }))
    }

    fn arguments(&self) -> serde_json::Value {
        serde_json::json!({"kind": "v05-experiment"})
    }

    fn on_model_loaded(&mut self, ctx: &ModelContext<'_>) -> Result<(), ExperimentError> {
        lock(&self.base).on_model_loaded(ctx)?;
        for co in &self.co {
            lock(co).on_model_loaded(ctx)?;
        }
        for (observer, _) in &self.observers {
            lock(observer).on_model_loaded(ctx)?;
        }
        Ok(())
    }

    fn before_prefill(&mut self, ctx: &ExecutionContext<'_>) -> Result<(), ExperimentError> {
        lock(&self.base).before_prefill(ctx)?;
        lock(&self.record).prompt = ctx.input_token_ids.unwrap_or_default().to_vec();
        for index in 0..self.co.len() {
            if let Err(error) = lock(&self.co[index]).before_prefill(ctx) {
                self.fail_co(index, error);
            }
        }
        for index in 0..self.observers.len() {
            let outcome = lock(&self.observers[index].0).before_prefill(ctx);
            if let Err(error) = outcome {
                self.fail_observer(index, error.message().to_string());
            }
        }
        Ok(())
    }

    fn before_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_layer(ctx, hidden, |experiment, ctx, tensor| {
            experiment.before_layer(ctx, tensor)
        })
    }

    fn after_attention(
        &mut self,
        ctx: &LayerContext<'_>,
        attention_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_layer(ctx, attention_output, |experiment, ctx, tensor| {
            experiment.after_attention(ctx, tensor)
        })
    }

    fn after_mlp(
        &mut self,
        ctx: &LayerContext<'_>,
        mlp_output: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_layer(ctx, mlp_output, |experiment, ctx, tensor| {
            experiment.after_mlp(ctx, tensor)
        })
    }

    fn after_layer(
        &mut self,
        ctx: &LayerContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_layer(ctx, hidden, |experiment, ctx, tensor| {
            experiment.after_layer(ctx, tensor)
        })?;
        // The residual stream entering block `layer + 1`, after every hook
        // of block `layer` (none of which mutated it before a boundary).
        if ctx.execution.phase == ExecutionPhase::Prefill
            && self.record_layers.contains(&(ctx.layer_index + 1))
        {
            let [rows, columns] = *hidden.shape();
            lock(&self.record).hidden.insert(
                ctx.layer_index + 1,
                CpuTensor::from_data(vec![rows, columns], hidden.values().to_vec()),
            );
        }
        Ok(())
    }

    fn before_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        hidden: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_logits(ctx, hidden, |experiment, ctx, tensor| {
            experiment.before_logits(ctx, tensor)
        })
    }

    fn after_logits(
        &mut self,
        ctx: &ExecutionContext<'_>,
        logits: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_logits(ctx, logits, |experiment, ctx, tensor| {
            experiment.after_logits(ctx, tensor)
        })
    }

    fn on_generation_complete(
        &mut self,
        ctx: &GenerationContext<'_>,
    ) -> Result<(), ExperimentError> {
        lock(&self.base).on_generation_complete(ctx)?;
        for index in 0..self.co.len() {
            if !self.co_failed(index)
                && let Err(error) = lock(&self.co[index]).on_generation_complete(ctx)
            {
                self.fail_co(index, error);
            }
        }
        Ok(())
    }
}

/// The recorded prefix of one base input.
struct InputPrefix {
    prompt: Vec<u32>,
    cache: Option<KVCache>,
    hidden: BTreeMap<usize, CpuTensor>,
}

/// How one input of a variant will run.
enum InputPlan {
    Resume {
        first_layer: usize,
        observations: PrefixObservations,
    },
    Full {
        reason: String,
    },
    /// Already consumed by `run_variant`.
    Taken,
}

/// Everything a base pass recorded for its variants.
pub(crate) struct SharedPrefix {
    inputs: Vec<InputPrefix>,
    /// `[variant][input]`.
    plans: Vec<Vec<InputPlan>>,
}

/// Result of a base pass: the base bundle, the co-baseline bundles (in
/// request order), and the recorded prefix for the variants.
pub(crate) struct BasePass {
    pub base: RunOutcome,
    pub co: Vec<RunOutcome>,
    pub prefix: SharedPrefix,
}

/// Run `f` inside a pool with `threads` workers (or directly when already
/// inside one of that size), as `execute_prepared` does.
fn in_pool<T: Send>(
    prepared: &mut PreparedRun,
    threads: usize,
    f: impl FnOnce(&mut PreparedRun) -> anyhow::Result<T> + Send,
) -> anyhow::Result<T> {
    if rayon::current_thread_index().is_some() && rayon::current_num_threads() == threads {
        return f(prepared);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("failed to build the experiment thread pool")?
        .install(move || f(prepared))
}

/// Run the base experiment once, with every eligible co-baseline observing
/// it and every variant's pre-boundary prefill observed, and write the
/// base and co-baseline bundles.
pub(crate) fn run_base_pass(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[&ExperimentSpecV1],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<BasePass> {
    ensure_same_session(prepared, base.resolved)?;
    for target in co {
        ensure_same_session(prepared, target.resolved)?;
    }
    for spec in variants {
        ensure_same_session(prepared, spec)?;
    }
    let threads = pool_threads(base.resolved)?;
    in_pool(prepared, threads, |prepared| {
        run_base_pass_inner(prepared, base, co, variants, cancel)
    })
}

fn run_base_pass_inner(
    prepared_mut: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[&ExperimentSpecV1],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<BasePass> {
    let prepared = &*prepared_mut;
    let n_layers = prepared.n_layers;
    let base_spec = base.resolved;
    let input_count = base_spec.inputs.len();

    let mut co_reasons: Vec<Option<String>> = co
        .iter()
        .map(|target| co_baseline_mismatch(base_spec, target.resolved, n_layers))
        .collect();
    let mut plans: Vec<Vec<InputPlan>> = variants
        .iter()
        .map(|_| Vec::with_capacity(input_count))
        .collect();
    let decisions: Vec<Vec<ReuseDecision>> = variants
        .iter()
        .map(|variant| {
            (0..variant.inputs.len())
                .map(|index| resume_decision(base_spec, variant, index, n_layers))
                .collect()
        })
        .collect();

    // Plans: every eligible co-baseline's, then the base's last so the
    // model runs the base plan (the co-baselines' decode plans are equal).
    let mut co_active: Vec<Option<ActiveSpec>> = Vec::with_capacity(co.len());
    for (target, reason) in co.iter().zip(&co_reasons) {
        co_active.push(if reason.is_none() {
            Some(activate_spec(prepared, target.resolved)?)
        } else {
            None
        });
    }
    let base_active = activate_spec(prepared, base_spec)?;

    let mut base_results: Vec<InputResult> = Vec::with_capacity(input_count);
    let mut co_results: Vec<Vec<InputResult>> = co.iter().map(|_| Vec::new()).collect();
    let mut timing = RunTiming::default();
    let mut inputs: Vec<InputPrefix> = Vec::with_capacity(input_count);

    for index in 0..input_count {
        let started = std::time::Instant::now();
        let base_experiment =
            new_input_experiment(prepared, base_spec, &base_active.bundle_sources, index)?;
        let mut live_co: Vec<(usize, Shared<V05Experiment>)> = Vec::new();
        for (co_index, target) in co.iter().enumerate() {
            if co_reasons[co_index].is_none() {
                live_co.push((
                    co_index,
                    new_input_experiment(prepared, target.resolved, &[], index)?,
                ));
            }
        }
        // Observers for the variants whose input `index` can resume.
        let mut observers: Vec<(usize, Shared<V05Experiment>, usize)> = Vec::new();
        for (variant_index, variant) in variants.iter().enumerate() {
            if let Some(ReuseDecision::Resume { first_layer }) = decisions[variant_index].get(index)
            {
                match new_input_experiment(prepared, variant, &[], index) {
                    Ok(observer) => observers.push((variant_index, observer, *first_layer)),
                    Err(error) => {
                        plans[variant_index].push(InputPlan::Full {
                            reason: format!("the variant's observer could not start: {error:#}"),
                        });
                    }
                }
            }
        }

        let record = Arc::new(Mutex::new(PassRecord {
            co_failures: vec![None; live_co.len()],
            observer_failures: vec![None; observers.len()],
            ..PassRecord::default()
        }));
        let pass = SharedPass {
            base: Arc::clone(&base_experiment),
            co: live_co.iter().map(|(_, co)| Arc::clone(co)).collect(),
            observers: observers
                .iter()
                .map(|(_, observer, boundary)| (Arc::clone(observer), *boundary))
                .collect(),
            record_layers: observers.iter().map(|(_, _, boundary)| *boundary).collect(),
            record: Arc::clone(&record),
        };
        let mut cache: Option<KVCache> = None;
        let role = (!observers.is_empty()).then_some(PrefixRole::Record { cache: &mut cache });
        let result = run_input(
            prepared,
            base_spec,
            &base_active,
            &base_experiment,
            Some(pass),
            role,
            cancel,
        )?;
        let elapsed = started.elapsed();
        timing.add(elapsed, &result);

        let mut record = lock(&record);
        for (slot, (co_index, co_experiment)) in live_co.iter().enumerate() {
            let collected = match &record.co_failures[slot] {
                Some(reason) => Err(reason.clone()),
                None => {
                    let mut experiment = lock(co_experiment);
                    experiment.set_generated_text(result.generated_text.clone());
                    experiment
                        .into_result()
                        .map_err(|error| error.message().to_string())
                }
            };
            match collected {
                Ok(co_result) => co_results[*co_index].push(co_result),
                Err(reason) => co_reasons[*co_index] = Some(format!("observer failed: {reason}")),
            }
        }
        for (slot, (variant_index, observer, boundary)) in observers.iter().enumerate() {
            let plan = match record.observer_failures[slot].take() {
                Some(reason) => InputPlan::Full {
                    reason: format!("the variant's observer stopped: {reason}"),
                },
                None => match lock(observer).take_prefix_observations() {
                    Ok(observations) => InputPlan::Resume {
                        first_layer: *boundary,
                        observations,
                    },
                    Err(error) => InputPlan::Full {
                        reason: error.message().to_string(),
                    },
                },
            };
            plans[*variant_index].push(plan);
        }
        // Variants that could not resume this input at all.
        for (variant_index, decision) in decisions.iter().enumerate() {
            if let Some(ReuseDecision::FullRecompute { reason }) = decision.get(index) {
                plans[variant_index].push(InputPlan::Full {
                    reason: reason.clone(),
                });
            }
        }
        inputs.push(InputPrefix {
            prompt: std::mem::take(&mut record.prompt),
            cache,
            hidden: std::mem::take(&mut record.hidden),
        });
        drop(record);
        base_results.push(result);
    }
    // A variant with more inputs than the base cannot share the extra ones.
    for (variant_index, variant) in variants.iter().enumerate() {
        while plans[variant_index].len() < variant.inputs.len() {
            plans[variant_index].push(InputPlan::Full {
                reason: "the base run has no identical input at this index".into(),
            });
        }
    }

    let served = variants.len();
    let co_shared = co_reasons.iter().filter(|reason| reason.is_none()).count();
    let base_outcome = finish_bundle(
        prepared,
        &base,
        &base_active,
        base_results,
        timing,
        Some(PrefixReuseRecord {
            role: "base",
            inputs: Vec::new(),
            note: Some(format!(
                "computed the shared prefix for {served} variant(s) and the generation for \
                 {co_shared} co-baseline(s)"
            )),
        }),
        cancel,
    )?;

    let mut co_outcomes = Vec::with_capacity(co.len());
    let prepared = prepared_mut;
    for (co_index, target) in co.iter().enumerate() {
        let outcome = match (&co_reasons[co_index], &co_active[co_index]) {
            (None, Some(active)) => finish_bundle(
                prepared,
                target,
                active,
                std::mem::take(&mut co_results[co_index]),
                timing,
                Some(PrefixReuseRecord {
                    role: "co-baseline",
                    inputs: Vec::new(),
                    note: Some(
                        "observed the base run's generation instead of running its own".into(),
                    ),
                }),
                cancel,
            )?,
            (reason, _) => {
                let reason = reason
                    .clone()
                    .unwrap_or_else(|| "no execution plan was built".into());
                let mut record = full_record(target.resolved, &reason);
                record.role = "co-baseline";
                run_full(prepared, target, record, cancel)?
            }
        };
        co_outcomes.push(outcome);
    }
    Ok(BasePass {
        base: base_outcome,
        co: co_outcomes,
        prefix: SharedPrefix { inputs, plans },
    })
}

fn full_record(spec: &ExperimentSpecV1, reason: &str) -> PrefixReuseRecord {
    PrefixReuseRecord {
        role: "variant",
        inputs: spec
            .inputs
            .iter()
            .map(|input| PrefixInputRecord {
                input_id: input.id.clone(),
                path: PrefixPath::FullRecompute {
                    reason: reason.to_string(),
                },
            })
            .collect(),
        note: None,
    }
}

/// Run a spec in full, in a pool of its own thread count.
fn run_full(
    prepared: &mut PreparedRun,
    target: &RunTarget<'_>,
    record: PrefixReuseRecord,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<RunOutcome> {
    in_pool(prepared, pool_threads(target.resolved)?, |prepared| {
        let prepared = &*prepared;
        let active = activate_spec(prepared, target.resolved)?;
        let mut results = Vec::new();
        let mut timing = RunTiming::default();
        for index in 0..target.resolved.inputs.len() {
            let started = std::time::Instant::now();
            let experiment =
                new_input_experiment(prepared, target.resolved, &active.bundle_sources, index)?;
            let result = run_input(
                prepared,
                target.resolved,
                &active,
                &experiment,
                None,
                None,
                cancel,
            )?;
            timing.add(started.elapsed(), &result);
            results.push(result);
        }
        finish_bundle(
            prepared,
            target,
            &active,
            results,
            timing,
            Some(record),
            cancel,
        )
    })
}

/// Run variant `variant_index` (as passed to [`run_base_pass`]) from the
/// recorded prefix and write its bundle. Each variant runs once.
pub(crate) fn run_variant(
    prepared: &mut PreparedRun,
    prefix: &mut SharedPrefix,
    variant_index: usize,
    target: RunTarget<'_>,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<RunOutcome> {
    ensure_same_session(prepared, target.resolved)?;
    let plans = prefix
        .plans
        .get_mut(variant_index)
        .ok_or_else(|| anyhow::anyhow!("variant {variant_index} was not part of the base pass"))?;
    let plans: Vec<InputPlan> = plans
        .iter_mut()
        .map(|plan| std::mem::replace(plan, InputPlan::Taken))
        .collect();
    if plans.iter().any(|plan| matches!(plan, InputPlan::Taken)) {
        anyhow::bail!("variant {variant_index} has already run");
    }
    let inputs = &prefix.inputs;
    in_pool(prepared, pool_threads(target.resolved)?, |prepared| {
        run_variant_inner(prepared, inputs, plans, &target, cancel)
    })
}

fn run_variant_inner(
    prepared: &PreparedRun,
    inputs: &[InputPrefix],
    plans: Vec<InputPlan>,
    target: &RunTarget<'_>,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<RunOutcome> {
    let spec = target.resolved;
    let active = activate_spec(prepared, spec)?;
    let mut results = Vec::with_capacity(spec.inputs.len());
    let mut records = Vec::with_capacity(spec.inputs.len());
    let mut timing = RunTiming::default();
    for (index, plan) in plans.into_iter().enumerate() {
        let started = std::time::Instant::now();
        let input_id = spec.inputs[index].id.clone();
        let experiment = new_input_experiment(prepared, spec, &active.bundle_sources, index)?;
        let resumable = match plan {
            InputPlan::Resume {
                first_layer,
                observations,
            } => match inputs.get(index) {
                Some(recorded) if recorded.prompt.len() < 2 => Err(
                    "a one-token prompt takes the single-token decode route, which has no \
                     layer boundary to resume at"
                        .to_string(),
                ),
                Some(InputPrefix {
                    prompt,
                    cache: Some(cache),
                    hidden,
                }) => match hidden.get(&first_layer) {
                    Some(hidden) => Ok((first_layer, observations, prompt, cache, hidden)),
                    None => Err("the base run did not record this boundary".to_string()),
                },
                _ => Err("the base run recorded no prefix for this input".to_string()),
            },
            InputPlan::Full { reason } => Err(reason),
            InputPlan::Taken => unreachable!("plans are taken once"),
        };
        let (result, path) = match resumable {
            Ok((first_layer, observations, prompt, cache, hidden)) => {
                lock(&experiment).inject_prefix_observations(observations);
                let mut outcome: Option<Result<usize, String>> = None;
                let role = PrefixRole::Resume {
                    first_layer,
                    prompt_token_ids: prompt,
                    hidden,
                    cache,
                    outcome: &mut outcome,
                };
                match run_input(
                    prepared,
                    spec,
                    &active,
                    &experiment,
                    None,
                    Some(role),
                    cancel,
                ) {
                    Ok(result) => match outcome {
                        Some(Ok(resume_layer)) => (result, PrefixPath::Resumed { resume_layer }),
                        _ => anyhow::bail!(
                            "input '{input_id}': the resumed run did not report its route"
                        ),
                    },
                    Err(error) => match outcome {
                        // Refused before computing anything: rerun it whole.
                        Some(Err(reason)) => {
                            let fresh = new_input_experiment(
                                prepared,
                                spec,
                                &active.bundle_sources,
                                index,
                            )?;
                            let result =
                                run_input(prepared, spec, &active, &fresh, None, None, cancel)?;
                            (result, PrefixPath::FullRecompute { reason })
                        }
                        _ => return Err(error),
                    },
                }
            }
            Err(reason) => {
                let result = run_input(prepared, spec, &active, &experiment, None, None, cancel)?;
                (result, PrefixPath::FullRecompute { reason })
            }
        };
        timing.add(started.elapsed(), &result);
        results.push(result);
        records.push(PrefixInputRecord { input_id, path });
    }
    finish_bundle(
        prepared,
        target,
        &active,
        results,
        timing,
        Some(PrefixReuseRecord {
            role: "variant",
            inputs: records,
            note: None,
        }),
        cancel,
    )
}

/// Base + co-baselines + variants in one call.
pub(crate) fn execute_shared(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<(RunOutcome, Vec<RunOutcome>, Vec<RunOutcome>)> {
    let specs: Vec<&ExperimentSpecV1> = variants.iter().map(|target| target.resolved).collect();
    let BasePass {
        base,
        co,
        mut prefix,
    } = run_base_pass(prepared, base, co, &specs, cancel)?;
    let mut outcomes = Vec::with_capacity(variants.len());
    for (index, target) in variants.iter().enumerate() {
        outcomes.push(run_variant(prepared, &mut prefix, index, *target, cancel)?);
    }
    Ok((base, co, outcomes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli_experiment::{execute_prepared, prepare_run};
    use crate::experiment_testutil::{out_dir, resolve, spec_text, tiny_model, TinyModel};
    use ember::quant_k::KStrategy;

    const INPUTS: &str = r#"
[[inputs]]
id = "a"
text = "w3 w17 w5 w40 w9 w22"

[[inputs]]
id = "b"
text = "w8 w1 w33 w2"
"#;

    /// Captures on both sides of every boundary the tests use, in every
    /// storage form, plus a generated-step capture.
    const CAPTURES: &str = r#"
[[captures]]
id = "final-rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "span-rows"
site = "residual-post-mlp"
layers = "all"
inputs = ["a"]
[captures.tokens]
kind = "matched-span"
text = "w17 w5"
occurrence = 0
subtokens = "final"

[[captures]]
id = "attn-summary"
site = "attention-output"
layers = [0, 1, 2]
storage = "summary-only"
[captures.tokens]
kind = "prompt-final"

[[captures]]
id = "pre-full"
site = "residual-pre-attention"
layers = [0, 2]
storage = "full-tensor"
[captures.tokens]
kind = "absolute-token"
index = 1

[[captures]]
id = "decode-mlp"
site = "mlp-output"
layers = [0, 3]
[captures.tokens]
kind = "generated-step"
step = 2

[[captures]]
id = "logits"
site = "logits"
[captures.tokens]
kind = "prompt-final"
"#;

    fn prefix_json(path: &std::path::Path) -> serde_json::Value {
        let text = std::fs::read_to_string(path.join("runtime.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["prefix_reuse"].clone()
    }

    /// Run base + co-baselines + variants once standalone and once shared;
    /// every bundle must be identical (semantic and payload hash). Returns
    /// each variant's recorded per-input paths.
    fn check(
        model: &TinyModel,
        mode: &str,
        temperature: f32,
        base_body: &str,
        co_bodies: &[&str],
        variant_bodies: &[&str],
    ) -> Vec<serde_json::Value> {
        let text = |body: &str| {
            spec_text(model, mode, 4, body).replace(
                "temperature = 0.0",
                &format!("temperature = {temperature:?}"),
            )
        };
        let base_text = text(base_body);
        let co_texts: Vec<String> = co_bodies.iter().map(|body| text(body)).collect();
        let variant_texts: Vec<String> = variant_bodies.iter().map(|body| text(body)).collect();
        let base_spec = resolve(&base_text);
        let co_specs: Vec<_> = co_texts.iter().map(|text| resolve(text)).collect();
        let variant_specs: Vec<_> = variant_texts.iter().map(|text| resolve(text)).collect();
        let mut prepared = prepare_run(&base_spec, KStrategy::Auto, false).unwrap();
        let dir = &model.dir;

        let mut full = Vec::new();
        for (index, (spec, text)) in std::iter::once((&base_spec, &base_text))
            .chain(co_specs.iter().zip(&co_texts))
            .chain(variant_specs.iter().zip(&variant_texts))
            .enumerate()
        {
            let (_, identity, report, _) = execute_prepared(
                &mut prepared,
                spec,
                text,
                &out_dir(dir, &format!("{mode}-{temperature}-full-{index}")),
                false,
                None,
            )
            .unwrap();
            assert!(report.ok);
            full.push(identity);
        }

        let name = |index: usize| out_dir(dir, &format!("{mode}-{temperature}-shared-{index}"));
        let base_out = name(0);
        let co_outs: Vec<_> = (0..co_specs.len()).map(|index| name(1 + index)).collect();
        let variant_outs: Vec<_> = (0..variant_specs.len())
            .map(|index| name(1 + co_specs.len() + index))
            .collect();
        let co_targets = targets(&co_specs, &co_texts, &co_outs);
        let variant_targets = targets(&variant_specs, &variant_texts, &variant_outs);
        let (base, co, variants) = execute_shared(
            &mut prepared,
            RunTarget {
                resolved: &base_spec,
                spec_text: &base_text,
                output_directory: &base_out,
                retain_incomplete: false,
            },
            &co_targets,
            &variant_targets,
            None,
        )
        .unwrap();
        let shared: Vec<&RunOutcome> = std::iter::once(&base)
            .chain(co.iter())
            .chain(variants.iter())
            .collect();
        for (index, (standalone, outcome)) in full.iter().zip(&shared).enumerate() {
            assert!(outcome.report.ok, "bundle {index} failed self-verification");
            assert_eq!(
                standalone.semantic_hash,
                outcome.identity.semantic_hash,
                "bundle {index} ({mode}): semantic hash differs from a standalone run; \
                 prefix record {}",
                prefix_json(&outcome.path)
            );
            assert_eq!(
                standalone.payload_hash, outcome.identity.payload_hash,
                "bundle {index} ({mode}): payload hash differs from a standalone run"
            );
        }
        assert_eq!(prefix_json(&base.path)["role"], "base");
        // The suite is only meaningful if interventions move the model.
        let moved = variants
            .iter()
            .filter(|outcome| {
                outcome.results.iter().zip(&base.results).any(|(a, b)| {
                    a.final_top1 != b.final_top1 || a.generated_token_ids != b.generated_token_ids
                })
            })
            .count();
        assert!(
            variants.is_empty() || moved > 0,
            "no variant changed the model's output"
        );
        variants
            .iter()
            .map(|outcome| prefix_json(&outcome.path)["inputs"].clone())
            .collect()
    }

    fn targets<'a>(
        specs: &'a [ExperimentSpecV1],
        texts: &'a [String],
        outs: &'a [std::path::PathBuf],
    ) -> Vec<RunTarget<'a>> {
        specs
            .iter()
            .zip(texts)
            .zip(outs)
            .map(|((resolved, spec_text), output_directory)| RunTarget {
                resolved,
                spec_text,
                output_directory,
                retain_incomplete: false,
            })
            .collect()
    }

    fn intervention(id: &str, site: &str, layers: &str, extra: &str, tokens: &str) -> String {
        format!(
            r#"
[[interventions]]
id = "{id}"
site = "{site}"
layers = {layers}
{extra}
[interventions.tokens]
{tokens}
"#
        )
    }

    fn paths(value: &serde_json::Value) -> Vec<(String, Option<u64>)> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|input| {
                (
                    input["path"].as_str().unwrap().to_string(),
                    input["resume_layer"].as_u64(),
                )
            })
            .collect()
    }

    const PROMPT_FINAL: &str = "kind = \"prompt-final\"";

    fn variant_suite(model: &TinyModel, mode: &str, temperature: f32) {
        let base = format!("{INPUTS}{CAPTURES}");
        let with = |parts: &[String]| format!("{base}{}", parts.concat());
        // Replace from a current-run capture (the morphology shape).
        let replace = with(&[intervention(
            "replace",
            "residual-post-mlp",
            "[2]",
            "operation = { kind = \"replace\" }\nsource = { kind = \"capture-from-current-run\", capture_id = \"final-rows\" }",
            PROMPT_FINAL,
        )]);
        // Zero an attention-output row mid-block, only for input b.
        let zero_b = with(&[intervention(
            "zero-b",
            "attention-output",
            "[1]",
            "operation = { kind = \"zero\" }\ninputs = [\"b\"]",
            "kind = \"relative-token\"\noffset_from_end = 1",
        )]);
        // Scale then restore exactly.
        let restore = with(&[
            intervention(
                "scale",
                "mlp-output",
                "[3]",
                "operation = { kind = \"scale\", factor = -2.0 }",
                PROMPT_FINAL,
            ),
            intervention(
                "restore",
                "mlp-output",
                "[3]",
                "operation = { kind = \"restore-original\" }",
                PROMPT_FINAL,
            ),
        ]);
        // Logits-site and decode-only interventions share every block.
        let logits = with(&[intervention(
            "scale-logits",
            "logits",
            "\"all\"",
            "operation = { kind = \"scale\", factor = 0.5 }",
            PROMPT_FINAL,
        )]);
        let decode_only = with(&[intervention(
            "decode",
            "residual-pre-attention",
            "[0]",
            "operation = { kind = \"zero\" }",
            "kind = \"generated-step\"\nstep = 2",
        )]);
        // Block 0 has no prefix: full recompute.
        let layer0 = with(&[intervention(
            "layer0",
            "residual-pre-attention",
            "[0]",
            "operation = { kind = \"scale\", factor = 3.0 }",
            PROMPT_FINAL,
        )]);
        let recorded = check(
            model,
            mode,
            temperature,
            &base,
            &[],
            &[&replace, &zero_b, &restore, &logits, &decode_only, &layer0],
        );
        let resumed = |layer| ("resumed".to_string(), Some(layer));
        let full = ("full-recompute".to_string(), None);
        let n = model.n_layers as u64;
        assert_eq!(paths(&recorded[0]), vec![resumed(2), resumed(2)]);
        assert_eq!(paths(&recorded[1]), vec![resumed(n), resumed(1)]);
        assert_eq!(paths(&recorded[2]), vec![resumed(3), resumed(3)]);
        assert_eq!(paths(&recorded[3]), vec![resumed(n), resumed(n)]);
        assert_eq!(paths(&recorded[4]), vec![resumed(n), resumed(n)]);
        assert_eq!(paths(&recorded[5]), vec![full.clone(), full]);
    }

    #[test]
    fn shared_prefix_bundles_equal_full_recompute_f32_reference() {
        let model = tiny_model("prefix-f32", 4, 64, false);
        variant_suite(&model, "reference", 0.0);
    }

    #[test]
    fn shared_prefix_bundles_equal_full_recompute_q8_reference_and_planned() {
        // Q8_0 weights take the fused single-token decode route.
        let model = tiny_model("prefix-q8", 4, 64, true);
        variant_suite(&model, "reference", 0.0);
        variant_suite(&model, "planned", 0.0);
        // Seeded sampling draws after the (shared) prefill.
        variant_suite(&model, "planned", 0.9);
    }

    #[test]
    fn one_token_prompts_and_mismatched_settings_fall_back_in_full() {
        let model = tiny_model("prefix-fallback", 3, 64, true);
        let base = r#"
[[inputs]]
id = "one"
text = "w5"

[[inputs]]
id = "two"
text = "w5 w6 w7"

[[captures]]
id = "rows"
site = "residual-post-mlp"
layers = "all"
[captures.tokens]
kind = "prompt-final"
"#;
        let variant = format!(
            "{base}{}",
            intervention(
                "zero",
                "residual-post-mlp",
                "[1]",
                "operation = { kind = \"zero\" }",
                PROMPT_FINAL,
            )
        );
        let recorded = check(&model, "reference", 0.0, base, &[], &[&variant]);
        let paths = paths(&recorded[0]);
        assert_eq!(paths[0].0, "full-recompute");
        assert_eq!(paths[1], ("resumed".to_string(), Some(1)));

        // Different generation settings: the spec-level check refuses.
        let base_spec = resolve(&spec_text(&model, "reference", 4, base));
        let longer = resolve(&spec_text(&model, "reference", 5, &variant));
        assert!(matches!(
            resume_decision(&base_spec, &longer, 1, model.n_layers),
            ReuseDecision::FullRecompute { .. }
        ));
    }

    /// The example morphology workflow (baseline capture, intervention,
    /// restoration) with the model resident: three standalone runs versus
    /// one shared pass, bundles compared by hash. Needs the pinned model in
    /// the repository root: `cargo test --release --bin ember
    /// example_workflow_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing harness; needs Llama-3.2-1B-Instruct-Q8_0.gguf"]
    fn example_workflow_timing() {
        let read = |name: &str| {
            let text =
                std::fs::read_to_string(format!("examples/experiments/{name}.toml")).unwrap();
            let spec = resolve(&text);
            (text, spec)
        };
        let (base_text, base_spec) = read("morphology-layerwise-capture");
        let variants = [
            read("morphology-intervention"),
            read("morphology-restoration"),
        ];
        if !base_spec.model.path.exists() {
            return;
        }
        let dir = crate::experiment_testutil::temp_dir("example-timing");
        let mut prepared = prepare_run(&base_spec, KStrategy::Auto, false).unwrap();
        let mut separate = Vec::new();
        let mut shared = Vec::new();
        for round in 0..4 {
            let out = |name: &str| dir.join(format!("{round}-{name}"));
            let started = std::time::Instant::now();
            let mut hashes = Vec::new();
            for (index, (text, spec)) in std::iter::once((&base_text, &base_spec))
                .chain(variants.iter().map(|(text, spec)| (text, spec)))
                .enumerate()
            {
                let (_, identity, _, _) = execute_prepared(
                    &mut prepared,
                    spec,
                    text,
                    &out(&format!("separate-{index}")),
                    false,
                    None,
                )
                .unwrap();
                hashes.push(identity.semantic_hash);
            }
            let separate_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = std::time::Instant::now();
            let outs: Vec<_> = (0..variants.len())
                .map(|index| out(&format!("shared-{}", index + 1)))
                .collect();
            let targets: Vec<RunTarget<'_>> = variants
                .iter()
                .zip(&outs)
                .map(|((text, spec), output_directory)| RunTarget {
                    resolved: spec,
                    spec_text: text,
                    output_directory,
                    retain_incomplete: false,
                })
                .collect();
            let base_out = out("shared-0");
            let (base, _, outcomes) = execute_shared(
                &mut prepared,
                RunTarget {
                    resolved: &base_spec,
                    spec_text: &base_text,
                    output_directory: &base_out,
                    retain_incomplete: false,
                },
                &[],
                &targets,
                None,
            )
            .unwrap();
            let shared_ms = started.elapsed().as_secs_f64() * 1000.0;
            let shared_hashes: Vec<String> = std::iter::once(&base)
                .chain(outcomes.iter())
                .map(|outcome| outcome.identity.semantic_hash.clone())
                .collect();
            assert_eq!(hashes, shared_hashes);
            if round > 0 {
                separate.push(separate_ms);
                shared.push(shared_ms);
            }
        }
        separate.sort_by(f64::total_cmp);
        shared.sort_by(f64::total_cmp);
        eprintln!(
            "morphology workflow (baseline + intervention + restoration, model resident): \
             separate {:.0} ms, shared prefix {:.0} ms (median of {})",
            separate[separate.len() / 2],
            shared[shared.len() / 2],
            separate.len()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn co_baselines_equal_their_standalone_bundles() {
        let model = tiny_model("prefix-co", 4, 64, true);
        let base = format!("{INPUTS}{CAPTURES}");
        let other = r#"
[[captures]]
id = "other"
site = "mlp-output"
layers = [1]
[captures.tokens]
kind = "prompt-final"
"#;
        let decode = r#"
[[captures]]
id = "decode-mlp"
site = "mlp-output"
layers = [0, 3]
[captures.tokens]
kind = "generated-step"
step = 2
"#;
        // Same generated-step sites as the base: observes the shared pass.
        let same_sites = format!("{INPUTS}{other}{decode}");
        // Different generated-step sites: runs on its own.
        let other_sites = format!("{INPUTS}{other}");
        let intervening = format!(
            "{base}{}",
            intervention(
                "zero",
                "residual-post-mlp",
                "[1]",
                "operation = { kind = \"zero\" }",
                PROMPT_FINAL,
            )
        );
        check(
            &model,
            "planned",
            0.0,
            &base,
            &[&same_sites, &other_sites],
            &[&intervening],
        );
    }
}
