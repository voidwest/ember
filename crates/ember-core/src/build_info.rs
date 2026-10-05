//! Build-time provenance recorded by this crate's build script: the commit
//! the binary was built from (and whether the tree was dirty), the compiler,
//! the target and the optimization level. The `ember` crate reads these
//! rather than running a second copy of the script, so both crates always
//! report the same build. `None` when the value was unavailable at build
//! time (no git, for instance).

pub const GIT_COMMIT: Option<&str> = option_env!("EMBER_GIT_COMMIT");
pub const GIT_DIRTY: Option<&str> = option_env!("EMBER_GIT_DIRTY");
pub const RUSTC_VERSION: Option<&str> = option_env!("EMBER_RUSTC_VERSION");
pub const TARGET: Option<&str> = option_env!("EMBER_TARGET");
pub const OPT_LEVEL: Option<&str> = option_env!("EMBER_OPT_LEVEL");
