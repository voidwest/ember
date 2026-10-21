//! Frozen sweep directories, rather than a current-writer round trip.
//!
//! `tests/fixtures/sweep-v1-plain` and `tests/fixtures/sweep-v1-effect` were
//! written on 2026-10-21 by `ember experiment run` (the attention-heads
//! branch on main 8226c84; sweep writing is unchanged there) on a synthetic
//! 2-layer model: a layer sweep of a `scale` intervention, the second with
//! `[sweep.effect]`. The model is not checked in; verification
//! reads only the bundles. A change that alters how `sweep.json` serializes,
//! what `verify` recalculates, or how an earlier field reads fails here.
use ember::v05::sweep::{compare_sweeps, read_manifest, verify_sweep};
use ember::v05::verify::VerifyOptions;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

const PLAIN_HASH: &str = "b7078353f2905bcae9ed607a657e84bdf00822b290aec0d919114861d1192d94";
const EFFECT_HASH: &str = "d24b62e9d190aaa44b47738bd911f240d626aaa72ad6543b0f511067bf03232e";

#[test]
fn frozen_sweeps_verify_with_their_recorded_hashes() {
    for (name, hash) in [
        ("sweep-v1-plain", PLAIN_HASH),
        ("sweep-v1-effect", EFFECT_HASH),
    ] {
        let options = VerifyOptions {
            expected_semantic_hash: Some(hash.to_string()),
            ..VerifyOptions::default()
        };
        let report = verify_sweep(&fixture(name), &options).unwrap();
        assert!(report.ok, "{name}: {:?}", report.checks);
        assert_eq!(report.sweep_hash, hash, "{name}");
        assert_eq!(report.bundles, 3, "{name}");
    }
}

#[test]
fn frozen_manifests_read_back_to_the_same_hash() {
    // An absent effect must not gain a field when the manifest is read and
    // hashed again.
    let plain = read_manifest(&fixture("sweep-v1-plain")).unwrap();
    assert!(plain.effect.is_none());
    assert!(plain.points.iter().all(|point| point.effect.is_none()));
    assert_eq!(plain.compute_hash(), PLAIN_HASH);
    let effect = read_manifest(&fixture("sweep-v1-effect")).unwrap();
    assert_eq!(effect.compute_hash(), EFFECT_HASH);
    assert!(effect.points.iter().all(|point| point.effect.is_some()));
    let verdict = compare_sweeps(&fixture("sweep-v1-plain"), &fixture("sweep-v1-plain"))
        .unwrap()
        .verdict;
    assert_eq!(verdict, "exact");
}
