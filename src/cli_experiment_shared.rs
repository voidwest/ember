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
//!
//! On models with the Q8_0 fast decode path the runs of a pass also decode
//! together ([`run_batched_pass`]): input by input, the base prefills, each
//! variant (and each co-baseline with a generation of its own) prefills from
//! the recorded prefix, and then every decode step of all of them is one
//! batched forward (`Llama::forward_decode_batch_with_experiments`), each
//! sequence keeping its own KV cache, hooks, sampler and stopping condition.
//! Batched decode is bit-identical per sequence to single decode, so the
//! bundles are too; `runtime.json` records `decode_batch` (the path and the
//! batch size). Passes the batch cannot express run sequentially and record
//! why ([`batch_ineligibility`]).

use crate::cli_experiment::{
    activate_spec, ensure_same_session, finish_bundle, finish_input, new_input_experiment,
    pool_threads, run_input, start_input, ActiveSpec, DecodeRoute, PreparedRun, RunOutcome,
    RunTarget, RunTiming,
};
use crate::cli_generation::{PrefixRole, SteppedGeneration};
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

    fn before_o_proj(
        &mut self,
        ctx: &LayerContext<'_>,
        heads: &mut TensorAccess<'_>,
    ) -> Result<(), ExperimentError> {
        self.fire_layer(ctx, heads, |experiment, ctx, tensor| {
            experiment.before_o_proj(ctx, tensor)
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
    /// Recorded in the variants' `runtime.json`.
    route: DecodeRoute,
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
    crate::cli_experiment::in_session_pool(threads, move || f(prepared))
}

/// Run the base experiment once, with every eligible co-baseline observing
/// it and every variant's pre-boundary prefill observed, and write the
/// base and co-baseline bundles.
///
/// `route` is what the bundles' `runtime.json` records as `decode_batch`.
pub(crate) fn run_base_pass(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[&ExperimentSpecV1],
    route: DecodeRoute,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<BasePass> {
    check_shared(prepared, base.resolved, co, variants)?;
    let threads = pool_threads(base.resolved)?;
    in_pool(prepared, threads, |prepared| {
        run_base_pass_inner(prepared, base, co, variants, route, cancel)
    })
}

/// The specs of a shared pass must share the session and declare no
/// analysis workflow.
fn check_shared(
    prepared: &PreparedRun,
    base: &ExperimentSpecV1,
    co: &[RunTarget<'_>],
    variants: &[&ExperimentSpecV1],
) -> anyhow::Result<()> {
    ensure_same_session(prepared, base)?;
    for spec in std::iter::once(base)
        .chain(co.iter().map(|target| target.resolved))
        .chain(variants.iter().copied())
    {
        anyhow::ensure!(
            spec.attribution.is_none() && spec.probe.is_none(),
            "experiment '{}' declares an analysis workflow ([attribution] or [probe]); run it \
             on its own, \
             not as part of a shared pass or sweep",
            spec.experiment.name
        );
    }
    for target in co {
        ensure_same_session(prepared, target.resolved)?;
    }
    for spec in variants {
        ensure_same_session(prepared, spec)?;
    }
    Ok(())
}

fn run_base_pass_inner(
    prepared_mut: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[&ExperimentSpecV1],
    route: DecodeRoute,
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
    let mut timing = RunTiming {
        decode: route,
        ..RunTiming::default()
    };
    let mut inputs: Vec<InputPrefix> = Vec::with_capacity(input_count);

    for index in 0..input_count {
        let started = std::time::Instant::now();
        let base_experiment = new_input_experiment(prepared, base_spec, Some(&base_active), index)?;
        let mut live_co: Vec<(usize, Shared<V05Experiment>)> = Vec::new();
        for (co_index, target) in co.iter().enumerate() {
            if co_reasons[co_index].is_none() {
                live_co.push((
                    co_index,
                    new_input_experiment(prepared, target.resolved, None, index)?,
                ));
            }
        }
        // Observers for the variants whose input `index` can resume.
        let mut observers: Vec<(usize, Shared<V05Experiment>, usize)> = Vec::new();
        for (variant_index, variant) in variants.iter().enumerate() {
            if let Some(ReuseDecision::Resume { first_layer }) = decisions[variant_index].get(index)
            {
                match new_input_experiment(prepared, variant, None, index) {
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
                run_full(prepared, target, record, route, cancel)?
            }
        };
        co_outcomes.push(outcome);
    }
    Ok(BasePass {
        base: base_outcome,
        co: co_outcomes,
        prefix: SharedPrefix {
            inputs,
            plans,
            route,
        },
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
    route: DecodeRoute,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<RunOutcome> {
    in_pool(prepared, pool_threads(target.resolved)?, |prepared| {
        let prepared = &*prepared;
        let active = activate_spec(prepared, target.resolved)?;
        let mut results = Vec::new();
        let mut timing = RunTiming {
            decode: route,
            ..RunTiming::default()
        };
        for index in 0..target.resolved.inputs.len() {
            let started = std::time::Instant::now();
            let experiment = new_input_experiment(prepared, target.resolved, Some(&active), index)?;
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
    let route = prefix.route;
    in_pool(prepared, pool_threads(target.resolved)?, |prepared| {
        run_variant_inner(prepared, inputs, plans, &target, route, cancel)
    })
}

fn run_variant_inner(
    prepared: &PreparedRun,
    inputs: &[InputPrefix],
    plans: Vec<InputPlan>,
    target: &RunTarget<'_>,
    route: DecodeRoute,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<RunOutcome> {
    let spec = target.resolved;
    let active = activate_spec(prepared, spec)?;
    let mut results = Vec::with_capacity(spec.inputs.len());
    let mut records = Vec::with_capacity(spec.inputs.len());
    let mut timing = RunTiming {
        decode: route,
        ..RunTiming::default()
    };
    for (index, plan) in plans.into_iter().enumerate() {
        let started = std::time::Instant::now();
        let input_id = spec.inputs[index].id.clone();
        let experiment = new_input_experiment(prepared, spec, Some(&active), index)?;
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
                            let fresh = new_input_experiment(prepared, spec, Some(&active), index)?;
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

/// Base + co-baselines + variants in one call: batched decode when the
/// pass allows it ([`batch_ineligibility`]), otherwise the sequential pass.
pub(crate) fn execute_shared(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<(RunOutcome, Vec<RunOutcome>, Vec<RunOutcome>)> {
    match batch_ineligibility(prepared, &base, co, variants)? {
        None => run_batched_pass(prepared, base, co, variants, cancel),
        Some(reason) => execute_shared_sequential(
            prepared,
            base,
            co,
            variants,
            DecodeRoute::Sequential { reason },
            cancel,
        ),
    }
}

/// [`execute_shared`] with every run decoding on its own.
pub(crate) fn execute_shared_sequential(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
    route: DecodeRoute,
    cancel: Option<&CancelToken>,
) -> anyhow::Result<(RunOutcome, Vec<RunOutcome>, Vec<RunOutcome>)> {
    let specs: Vec<&ExperimentSpecV1> = variants.iter().map(|target| target.resolved).collect();
    let BasePass {
        base,
        co,
        mut prefix,
    } = run_base_pass(prepared, base, co, &specs, route, cancel)?;
    let mut outcomes = Vec::with_capacity(variants.len());
    for (index, target) in variants.iter().enumerate() {
        outcomes.push(run_variant(prepared, &mut prefix, index, *target, cancel)?);
    }
    Ok((base, co, outcomes))
}

// ---------------------------------------------------------------------------
// batched decode
// ---------------------------------------------------------------------------

/// Why the runs of a shared pass cannot decode as one batch, or `None` when
/// they can. Anything the batch cannot express keeps the sequential pass.
pub(crate) fn batch_ineligibility(
    prepared: &PreparedRun,
    base: &RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
) -> anyhow::Result<Option<&'static str>> {
    if co.is_empty() && variants.is_empty() {
        return Ok(Some("the pass has a single generation"));
    }
    if let Some(reason) = crate::cli_generation::batched_decode_ineligibility(&prepared.model) {
        return Ok(Some(reason));
    }
    let threads = pool_threads(base.resolved)?;
    let mode = base.resolved.execution.mode;
    for target in co.iter().chain(variants) {
        if pool_threads(target.resolved)? != threads {
            return Ok(Some("the runs request different thread counts"));
        }
        if target.resolved.execution.mode != mode {
            return Ok(Some("the runs use different execution modes"));
        }
    }
    Ok(None)
}

/// A participant of one input's joint decode.
enum Seat {
    Base,
    /// A co-baseline that runs its own generation (index into `co`).
    Co(usize),
    Variant(usize, PrefixPath),
}

/// Base + co-baselines + variants with every generation of an input decoded
/// in one batch (see the module docs). Bundles are bit-identical to the
/// sequential pass; `runtime.json` records `decode_batch`.
///
/// Call only when [`batch_ineligibility`] returns `None`.
pub(crate) fn run_batched_pass(
    prepared: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<(RunOutcome, Vec<RunOutcome>, Vec<RunOutcome>)> {
    let specs: Vec<&ExperimentSpecV1> = variants.iter().map(|target| target.resolved).collect();
    check_shared(prepared, base.resolved, co, &specs)?;
    let threads = pool_threads(base.resolved)?;
    in_pool(prepared, threads, |prepared| {
        run_batched_inner(prepared, base, co, variants, cancel)
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one sequential batched-run flow whose phases share local state"
)]
fn run_batched_inner(
    prepared_mut: &mut PreparedRun,
    base: RunTarget<'_>,
    co: &[RunTarget<'_>],
    variants: &[RunTarget<'_>],
    cancel: Option<&CancelToken>,
) -> anyhow::Result<(RunOutcome, Vec<RunOutcome>, Vec<RunOutcome>)> {
    let prepared = &*prepared_mut;
    let backend = ember::backend::CpuBackend;
    let n_layers = prepared.n_layers;
    let base_spec = base.resolved;

    // Co-baselines that cannot observe the base run their own generation
    // (in the batch); the rest observe the base's.
    let mut co_reasons: Vec<Option<String>> = co
        .iter()
        .map(|target| co_baseline_mismatch(base_spec, target.resolved, n_layers))
        .collect();
    let co_own: Vec<bool> = co_reasons.iter().map(Option::is_some).collect();
    let decisions: Vec<Vec<ReuseDecision>> = variants
        .iter()
        .map(|variant| {
            (0..variant.resolved.inputs.len())
                .map(|index| resume_decision(base_spec, variant.resolved, index, n_layers))
                .collect()
        })
        .collect();

    // Every spec's plan and sources; the base's last so the model is left
    // in its mode (all modes are equal here, and the Q8_0 fast decode the
    // batch reproduces does not consult the plan).
    let mut co_active = Vec::with_capacity(co.len());
    for target in co {
        co_active.push(activate_spec(prepared, target.resolved)?);
    }
    let mut variant_active = Vec::with_capacity(variants.len());
    for target in variants {
        variant_active.push(activate_spec(prepared, target.resolved)?);
    }
    let base_active = activate_spec(prepared, base_spec)?;

    let mut base_results: Vec<InputResult> = Vec::with_capacity(base_spec.inputs.len());
    let mut co_results: Vec<Vec<InputResult>> = co.iter().map(|_| Vec::new()).collect();
    let mut variant_results: Vec<Vec<InputResult>> = variants.iter().map(|_| Vec::new()).collect();
    let mut variant_records: Vec<Vec<PrefixInputRecord>> =
        variants.iter().map(|_| Vec::new()).collect();
    let mut base_timing = RunTiming::default();
    let mut co_timing: Vec<RunTiming> = co.iter().map(|_| RunTiming::default()).collect();
    let mut variant_timing: Vec<RunTiming> =
        variants.iter().map(|_| RunTiming::default()).collect();
    let mut batch_size = 0usize;

    let input_count = std::iter::once(base_spec.inputs.len())
        .chain(
            co.iter()
                .zip(&co_own)
                .filter(|(_, own)| **own)
                .map(|(target, _)| target.resolved.inputs.len()),
        )
        .chain(variants.iter().map(|target| target.resolved.inputs.len()))
        .max()
        .unwrap_or(0);
    for index in 0..input_count {
        let started = std::time::Instant::now();
        let mut seats: Vec<(Seat, Shared<V05Experiment>, SteppedGeneration<'_>)> = Vec::new();

        // -- the base, with its co-baselines and variant observers --
        let mut base_pass = None;
        let mut prefix: Option<InputPrefix> = None;
        let mut plans: Vec<Option<InputPlan>> = variants.iter().map(|_| None).collect();
        if index < base_spec.inputs.len() {
            let base_experiment =
                new_input_experiment(prepared, base_spec, Some(&base_active), index)?;
            let mut live_co: Vec<(usize, Shared<V05Experiment>)> = Vec::new();
            for (co_index, target) in co.iter().enumerate() {
                if co_reasons[co_index].is_none() {
                    live_co.push((
                        co_index,
                        new_input_experiment(prepared, target.resolved, None, index)?,
                    ));
                }
            }
            let mut observers: Vec<(usize, Shared<V05Experiment>, usize)> = Vec::new();
            for (variant_index, target) in variants.iter().enumerate() {
                if let Some(ReuseDecision::Resume { first_layer }) =
                    decisions[variant_index].get(index)
                {
                    match new_input_experiment(prepared, target.resolved, None, index) {
                        Ok(observer) => observers.push((variant_index, observer, *first_layer)),
                        Err(error) => {
                            plans[variant_index] = Some(InputPlan::Full {
                                reason: format!(
                                    "the variant's observer could not start: {error:#}"
                                ),
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
            let generation = start_input(
                prepared,
                base_spec,
                &base_experiment,
                Some(pass),
                role,
                cancel,
            )?;
            // Observers only see the prefill, which is complete now.
            let mut record_guard = lock(&record);
            for (slot, (variant_index, observer, boundary)) in observers.iter().enumerate() {
                let plan = match record_guard.observer_failures[slot].take() {
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
                plans[*variant_index] = Some(plan);
            }
            prefix = Some(InputPrefix {
                prompt: std::mem::take(&mut record_guard.prompt),
                cache,
                hidden: std::mem::take(&mut record_guard.hidden),
            });
            drop(record_guard);
            base_pass = Some((live_co, record));
            seats.push((Seat::Base, base_experiment, generation));
        }

        // -- co-baselines with a generation of their own --
        for (co_index, target) in co.iter().enumerate() {
            if co_own[co_index] && index < target.resolved.inputs.len() {
                let experiment = new_input_experiment(
                    prepared,
                    target.resolved,
                    Some(&co_active[co_index]),
                    index,
                )?;
                let generation =
                    start_input(prepared, target.resolved, &experiment, None, None, cancel)?;
                seats.push((Seat::Co(co_index), experiment, generation));
            }
        }

        // -- variants, resumed from the base's prefix where possible --
        for (variant_index, target) in variants.iter().enumerate() {
            let spec = target.resolved;
            if index >= spec.inputs.len() {
                continue;
            }
            let active = &variant_active[variant_index];
            // The reasons the sequential pass records, in its precedence.
            let plan = match (
                plans[variant_index].take(),
                decisions[variant_index].get(index),
            ) {
                _ if index >= base_spec.inputs.len() => InputPlan::Full {
                    reason: "the base run has no identical input at this index".into(),
                },
                (Some(plan), _) => plan,
                (None, Some(ReuseDecision::FullRecompute { reason })) => InputPlan::Full {
                    reason: reason.clone(),
                },
                (None, _) => InputPlan::Full {
                    reason: "the base run has no identical input at this index".into(),
                },
            };
            let experiment = new_input_experiment(prepared, spec, Some(active), index)?;
            let resumable = match plan {
                InputPlan::Resume {
                    first_layer,
                    observations,
                } => match &prefix {
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
            let (experiment, generation, path) = match resumable {
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
                    match start_input(prepared, spec, &experiment, None, Some(role), cancel) {
                        Ok(generation) => match outcome {
                            Some(Ok(resume_layer)) => {
                                (experiment, generation, PrefixPath::Resumed { resume_layer })
                            }
                            _ => anyhow::bail!(
                                "input '{}': the resumed run did not report its route",
                                spec.inputs[index].id
                            ),
                        },
                        Err(error) => match outcome {
                            // Refused before computing anything: rerun it whole.
                            Some(Err(reason)) => {
                                let fresh =
                                    new_input_experiment(prepared, spec, Some(active), index)?;
                                let generation =
                                    start_input(prepared, spec, &fresh, None, None, cancel)?;
                                (fresh, generation, PrefixPath::FullRecompute { reason })
                            }
                            _ => return Err(error),
                        },
                    }
                }
                Err(reason) => {
                    let generation = start_input(prepared, spec, &experiment, None, None, cancel)?;
                    (experiment, generation, PrefixPath::FullRecompute { reason })
                }
            };
            seats.push((Seat::Variant(variant_index, path), experiment, generation));
        }
        drop(prefix);

        // -- one batch for every generation of this input --
        let mut generations: Vec<&mut SteppedGeneration<'_>> = seats
            .iter_mut()
            .map(|(_, _, generation)| generation)
            .collect();
        batch_size = batch_size.max(crate::cli_generation::decode_together(
            &backend,
            &prepared.model,
            &prepared.tokenizer,
            &mut generations,
            cancel,
        )?);
        drop(generations);

        for (seat, experiment, generation) in seats {
            let result = finish_input(prepared, &experiment, generation)?;
            // Every generation of the input shares its wall time.
            let elapsed = started.elapsed();
            match &seat {
                Seat::Base => base_timing.add(elapsed, &result),
                Seat::Co(co_index) => co_timing[*co_index].add(elapsed, &result),
                Seat::Variant(variant_index, _) => {
                    variant_timing[*variant_index].add(elapsed, &result)
                }
            }
            match seat {
                Seat::Base => {
                    let (live_co, record) = base_pass.take().expect("the base ran this input");
                    let record = lock(&record);
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
                            Err(reason) => {
                                co_reasons[*co_index] = Some(format!("observer failed: {reason}"));
                            }
                        }
                    }
                    base_results.push(result);
                }
                Seat::Co(co_index) => co_results[co_index].push(result),
                Seat::Variant(variant_index, path) => {
                    variant_records[variant_index].push(PrefixInputRecord {
                        input_id: variants[variant_index].resolved.inputs[index].id.clone(),
                        path,
                    });
                    variant_results[variant_index].push(result);
                }
            }
        }
    }

    // A cancel that landed after the last decode step still leaves nothing
    // behind; past this point every bundle is written.
    if cancel.is_some_and(CancelToken::is_cancelled) {
        return Err(anyhow::Error::new(ember::cancel::Cancelled));
    }
    let generations = 1 + co_own.iter().filter(|own| **own).count() + variants.len();
    let route = DecodeRoute::Batched {
        batch_size,
        generations,
    };
    base_timing.decode = route;
    let served = variants.len();
    let co_shared = co_reasons.iter().filter(|reason| reason.is_none()).count();
    let base_outcome = finish_bundle(
        prepared,
        &base,
        &base_active,
        base_results,
        base_timing,
        Some(PrefixReuseRecord {
            role: "base",
            inputs: Vec::new(),
            note: Some(format!(
                "computed the shared prefix for {served} variant(s) and the generation for \
                 {co_shared} co-baseline(s)"
            )),
        }),
        None,
    )?;
    let mut co_outcomes = Vec::with_capacity(co.len());
    let mut refused = Vec::new();
    for (co_index, target) in co.iter().enumerate() {
        let record = match (&co_reasons[co_index], co_own[co_index]) {
            (None, _) => PrefixReuseRecord {
                role: "co-baseline",
                inputs: Vec::new(),
                note: Some("observed the base run's generation instead of running its own".into()),
            },
            (Some(reason), true) => {
                let mut record = full_record(target.resolved, reason);
                record.role = "co-baseline";
                record
            }
            (Some(_), false) => {
                // Its observer failed mid-generation: run it on its own below.
                refused.push(co_index);
                co_outcomes.push(None);
                continue;
            }
        };
        co_outcomes.push(Some(finish_bundle(
            prepared,
            target,
            &co_active[co_index],
            std::mem::take(&mut co_results[co_index]),
            if co_own[co_index] {
                RunTiming {
                    decode: route,
                    ..co_timing[co_index]
                }
            } else {
                base_timing
            },
            Some(record),
            None,
        )?));
    }
    let mut variant_outcomes = Vec::with_capacity(variants.len());
    for (variant_index, target) in variants.iter().enumerate() {
        variant_outcomes.push(finish_bundle(
            prepared,
            target,
            &variant_active[variant_index],
            std::mem::take(&mut variant_results[variant_index]),
            RunTiming {
                decode: route,
                ..variant_timing[variant_index]
            },
            Some(PrefixReuseRecord {
                role: "variant",
                inputs: std::mem::take(&mut variant_records[variant_index]),
                note: None,
            }),
            None,
        )?);
    }
    let prepared = prepared_mut;
    for co_index in refused {
        let target = &co[co_index];
        let reason = co_reasons[co_index].clone().unwrap_or_default();
        let mut record = full_record(target.resolved, &reason);
        record.role = "co-baseline";
        co_outcomes[co_index] = Some(run_full(
            prepared,
            target,
            record,
            DecodeRoute::Sequential {
                reason: "the co-baseline's observer failed during the batched generation",
            },
            cancel,
        )?);
    }
    Ok((
        base_outcome,
        co_outcomes
            .into_iter()
            .map(|outcome| outcome.expect("every co-baseline ran"))
            .collect(),
        variant_outcomes,
    ))
}

#[cfg(test)]
mod tests;
