//! Small, embedded line-icon set for the native console.

use gpui::{prelude::*, svg, AssetSource, Result, SharedString, Svg};
use std::borrow::Cow;

macro_rules! icons {
    ($(($name:ident, $file:literal)),+ $(,)?) => {
        $(pub(super) const $name: &str = concat!("ember-icons/", $file, ".svg");)+

        pub(super) struct Assets;

        impl AssetSource for Assets {
            fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
                Ok(match path {
                    $(concat!("ember-icons/", $file, ".svg") => Some(Cow::Borrowed(
                        include_bytes!(concat!("../gui_icons/", $file, ".svg")),
                    )),)+
                    _ => None,
                })
            }

            fn list(&self, path: &str) -> Result<Vec<SharedString>> {
                let paths = [$(concat!("ember-icons/", $file, ".svg")),+];
                Ok(paths
                    .into_iter()
                    .filter(|candidate| candidate.starts_with(path))
                    .map(SharedString::from)
                    .collect())
            }
        }
    };
}

icons![
    (CHEVRON_DOWN, "chevron-down"),
    (MOON, "moon"),
    (SUN, "sun"),
    (MONITOR, "monitor"),
    (PLAY, "play"),
    (RESTORE, "restore"),
    (MODEL, "model"),
    (CHECK, "check"),
    (WARNING, "warning"),
    (COPY, "copy"),
];

pub(super) fn icon(path: &'static str) -> Svg {
    svg().path(path).flex_none()
}
