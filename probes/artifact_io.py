"""Shared artifact writes and hashes without importing the probe training stack."""

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _atomic_file(path: str | Path, *, text: bool = False):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        options = {"encoding": "utf-8", "newline": "\n"} if text else {}
        with os.fdopen(descriptor, "w" if text else "wb", **options) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_savez(path: str | Path, **arrays) -> None:
    import numpy as np

    with _atomic_file(path) as handle:
        np.savez(handle, **arrays)


def atomic_save_npy(path: str | Path, array) -> None:
    import numpy as np

    with _atomic_file(path) as handle:
        np.save(handle, array, allow_pickle=False)


def atomic_write_text(path: str | Path, content: str) -> None:
    with _atomic_file(path, text=True) as handle:
        handle.write(content)


def atomic_save_figure(figure, path: str | Path, **savefig_kwargs) -> None:
    """Preserve the filename extension used by the figure's format selection."""
    output = Path(path)
    if not output.suffix:
        raise ValueError("figure output path requires a file extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, **savefig_kwargs)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
