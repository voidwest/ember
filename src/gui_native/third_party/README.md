# Third-party code in the console

Parts of the console are adapted from **Ely GPUI Components**
(https://github.com/ZacharyZhang-NY/Ely-GPUI-Components), dual-licensed MIT OR
Apache-2.0. The MIT licence text is in `LICENSE-ELY-MIT`. Ely is built on
upstream Zed GPUI and Ember on gpui-kit's fork, so nothing is linked: the code
is copied and adapted (theme lookups replaced by Ember's colours, APIs mapped to
gpui-kit). Each adapted file names its source at the top.

| Ember file | Adapted from Ely |
|---|---|
| `spark.rs` | `src/data_display/spark.rs` |
| `skeleton.rs` | `src/motion/skeleton.rs` (`Skeleton`, `SkeletonText`) |
| `chart.rs` (`svg_export`) | `src/charts/export.rs` (`hex`, `escaped`, SVG page layout) |
| `views.rs` (toast) | the behaviour of `src/feedback/toast.rs` (a timed, bottom-right confirmation), reimplemented small |
