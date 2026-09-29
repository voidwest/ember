//! Unit tests for the console.

use super::{theme, truncate_chars, AppearanceMode, FormValues};
use crate::gui::parse_run_request;

#[test]
fn unreadable_store_is_never_written_back() {
    let dir = std::env::temp_dir().join(format!("ember-open-store-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("app-state.v2.json");
    let legacy = dir.join("app-state.json");
    std::fs::write(&path, b"{ not a store").unwrap();

    let (store, error, write_path) = super::open_store(path.clone(), &legacy);
    assert!(store.runs.is_empty());
    assert!(error.unwrap().contains("left untouched"));
    assert!(
        write_path.is_none(),
        "a damaged store must not be overwritten"
    );

    // A readable store is written back to the same place.
    super::AppStore::default().write(&path).unwrap();
    let (_, error, write_path) = super::open_store(path.clone(), &legacy);
    assert!(error.is_none());
    assert_eq!(write_path.as_deref(), Some(path.as_path()));

    // A damaged legacy file is reported, left alone, and does not stop this
    // build's own store from being written: it never writes the legacy file.
    std::fs::write(&legacy, b"{ old and damaged").unwrap();
    let (_, error, write_path) = super::open_store(path.clone(), &legacy);
    assert!(error.unwrap().contains("older Ember"));
    assert_eq!(write_path.as_deref(), Some(path.as_path()));
    assert_eq!(std::fs::read(&legacy).unwrap(), b"{ old and damaged");
    let _ = std::fs::remove_dir_all(&dir);
}

fn form() -> FormValues {
    FormValues {
        model_path: "model.gguf".to_string(),
        prompt: "اختبار".to_string(),
        max_tokens: "48".to_string(),
        execution: "reference".to_string(),
        site: "after-mlp".to_string(),
        layer: "8".to_string(),
        op: "scale".to_string(),
        value: "0.5".to_string(),
        source: "capture".to_string(),
        source_layer: "0".to_string(),
        token: "prompt-final".to_string(),
        span: String::new(),
    }
}

#[test]
fn appearance_mode_cycles_and_resolves_system_theme() {
    assert_eq!(AppearanceMode::System.next(), AppearanceMode::Dark);
    assert_eq!(AppearanceMode::Dark.next(), AppearanceMode::Light);
    assert_eq!(AppearanceMode::Light.next(), AppearanceMode::System);
    assert!(AppearanceMode::System.is_dark(true));
    assert!(!AppearanceMode::System.is_dark(false));
    assert!(AppearanceMode::Dark.is_dark(false));
    assert!(!AppearanceMode::Light.is_dark(true));
}

#[test]
fn light_and_dark_palettes_differ_in_core_semantic_roles() {
    let dark = theme::dark();
    let light = theme::light();
    assert_ne!(dark.canvas, light.canvas);
    assert_ne!(dark.sidebar, light.sidebar);
    assert_ne!(dark.surface, light.surface);
    assert_ne!(dark.text, light.text);
    assert_ne!(dark.text_muted, light.text_muted);
    assert_ne!(dark.border, light.border);
    assert_ne!(dark.accent, light.accent);
    assert_ne!(dark.ok, light.ok);
    assert_ne!(dark.err, light.err);
    assert_ne!(dark.warn, light.warn);
    assert_ne!(dark.err_box_bg, light.err_box_bg);
}

#[test]
fn default_form_builds_valid_run_request() {
    let request = form().build_run_request().unwrap();
    let config = parse_run_request(&request).expect("default config validates");
    assert_eq!(config.site.stage_id(), "after-mlp");
    assert_eq!(config.layer, Some(8));
    assert!(matches!(config.operation, crate::gui::GuiOperation::Scale));
}

#[test]
fn non_per_layer_site_drops_layer() {
    let mut form = form();
    form.site = "before-logits".to_string();
    assert!(form.build_run_request().unwrap().layer.is_none());
}

#[test]
fn scale_factor_parses_and_validates() {
    let mut form = form();
    form.value = "abc".to_string();
    assert!(form.build_run_request().is_err());
    form.value = "0.25".to_string();
    let config = parse_run_request(&form.build_run_request().unwrap()).unwrap();
    assert_eq!(config.factor, 0.25);
}

#[test]
fn source_layer_is_clamped_below_target() {
    let mut form = form();
    form.layer = "7".to_string();
    form.source_layer = "9".to_string();
    form.op = "replace".to_string();
    let config = parse_run_request(&form.build_run_request().unwrap()).unwrap();
    assert_eq!(config.source_layer, Some(6));
}

#[test]
#[ignore = "requires EMBER_GUI_TEST_MODEL; writes verified experiment bundles"]
fn native_worker_runs_and_restores_real_model() {
    use super::{spawn_worker, WorkerMsg, WorkerReply};
    use ember::quant_k::KStrategy;
    use std::time::Duration;
    let model = std::env::var("EMBER_GUI_TEST_MODEL").expect("set EMBER_GUI_TEST_MODEL");
    let mut values = form();
    values.model_path = model.clone();
    values.prompt = "The capital of France is".into();
    values.max_tokens = "4".into();
    let mut request = values.build_run_request().unwrap();
    let config = parse_run_request(&request).unwrap();
    let (tx, rx) = spawn_worker(KStrategy::Auto, false);
    let receive = || {
        rx.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(300))
            .unwrap()
    };
    tx.send(WorkerMsg::Prepare(model)).unwrap();
    let WorkerReply::Prepared(info) = receive() else {
        panic!("expected prepared reply")
    };
    let info = info.unwrap();
    assert!(info.n_layers > 8);
    tx.send(WorkerMsg::Run(config)).unwrap();
    let WorkerReply::RunDone(result) = receive() else {
        panic!("expected run reply")
    };
    let run = result.unwrap();
    assert!(run.verification.ok);
    assert!(!run.baseline.generated_token_ids.is_empty());
    println!(
        "baseline: {}\nintervention: {}",
        run.baseline.text, run.intervention.text
    );
    println!(
        "baseline bundle: {}\nintervention bundle: {}",
        run.baseline.bundle_dir, run.intervention.bundle_dir
    );
    request.operation = "restore-original".into();
    request.factor = None;
    request.alpha = None;
    request.source_layer = None;
    tx.send(WorkerMsg::Restore(parse_run_request(&request).unwrap()))
        .unwrap();
    let WorkerReply::RestoreDone(result) = receive() else {
        panic!("expected restore reply")
    };
    let restored = result.unwrap();
    assert!(restored.verification.ok);
    assert!(restored.baseline_comparable);
    assert!(restored.matches_baseline);
    assert_eq!(
        restored.output.generated_token_ids,
        run.baseline.generated_token_ids
    );
    println!("restoration bundle: {}", restored.output.bundle_dir);
}

#[test]
fn inspector_excerpt_preserves_arabic_characters() {
    assert_eq!(truncate_chars("المدينة المنورة", 7), "المدينة…");
    assert_eq!(truncate_chars("اختبار", 20), "اختبار");
}

#[test]
fn a_store_from_a_newer_build_opens_read_only() {
    let dir = std::env::temp_dir().join(format!("ember-open-newer-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("app-state.v2.json");
    let mut newer = super::AppStore::default();
    newer.schema_minor = super::app_store::STORE_SCHEMA_MINOR + 1;
    std::fs::write(&path, serde_json::to_vec(&newer).unwrap()).unwrap();

    let (store, error, write_path) = super::open_store(path.clone(), &dir.join("app-state.json"));
    assert!(store.written_by_newer_build(), "the store is still shown");
    assert!(error.unwrap().contains("newer Ember"));
    assert!(write_path.is_none(), "and never written back");
    let _ = std::fs::remove_dir_all(&dir);
}
