fn main() {
    // A Python extension module resolves the interpreter's symbols at import
    // time. maturin passes the linker flags for that itself; this makes a
    // plain `cargo build -p ember-python` link on macOS too.
    pyo3_build_config::add_extension_module_link_args();
}
