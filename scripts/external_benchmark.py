#!/usr/bin/env python3
"""Run a fair, command-driven benchmark matrix without bundling model files.

The input is a JSON specification (``ember.external-benchmark.v1``). Every
runtime is an argv vector, never a shell string, and is run in fresh processes
for each case/repetition. The harness records captured stdout/stderr bytes
(exact for complete trials and capped per-stream prefixes for output-limit failures),
wall/CPU/resource observations, and pairwise output/timing comparisons. It
does not designate a reference runtime or judge output correctness.

Example::

    python3 scripts/external_benchmark.py \\
        --spec benchmark.json --output runs/external/2026-08-28

The JSON schema and submission guidance live in ``docs/external-benchmark.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import stat as stat_module

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module.
    resource = None  # type: ignore[assignment]
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "ember.external-benchmark.v1"
SCRIPT_PATH = Path(__file__).resolve()
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
# Preflight caps keep a malformed specification from turning the runner into
# an unbounded scheduler or reader. They are deliberately far above the
# default matrix, but finite because specs may come from automation.
MAX_SPEC_BYTES = 8 * 1024 * 1024
MAX_ID_CHARS = 128
MAX_JSON_DEPTH = 128
MAX_RUNTIMES = 64
MAX_CASES = 256
MAX_WARMUPS = 1_000
MAX_REPETITIONS = 1_000
MAX_TOTAL_TRIALS = 10_000
MAX_TIMEOUT_SECONDS = 6 * 60 * 60
MAX_OUTPUT_BYTES = 256 << 20
# Aggregate declarations are bounded too: otherwise a matrix at the per-trial
# maxima could reserve an impractical amount of host time or disk space.
MAX_TOTAL_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
MAX_TOTAL_OUTPUT_BYTES = 8 << 30
MAX_OUTPUT_FILES = 100_000
MAX_OUTPUT_ENTRIES = 200_000
MAX_OUTPUT_PATH_DEPTH = 32
# Hashing is also a file-input boundary (the output tree and executable path
# are user-controlled); never stream an unbounded regular file.
MAX_HASH_BYTES = MAX_OUTPUT_BYTES
POLL_SECONDS = 0.01


class SpecError(ValueError):
    """A user-supplied benchmark specification is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def canonical_json(value: Any) -> bytes:
    """Encode JSON in the stable form used for all specification hashes."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        # Keep this inside the guarded region: JSON permits escaped lone
        # surrogates, which cannot be emitted as UTF-8 with ensure_ascii=False.
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SpecError(f"spec is not canonical JSON: {error}") from error


def pretty_json(value: Any) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
        return (text + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SpecError(f"value is not JSON serializable: {error}") from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.lstat()
    if not stat_module.S_ISREG(stat.st_mode):
        raise OSError(f"not a regular file: {path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def open_regular_file(path: Path) -> Any:
    """Open a regular file without following a replacement symlink."""
    before = file_identity(path)
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if nonblock:
        flags |= nonblock
    fd = os.open(path, flags)
    try:
        after = os.fstat(fd)
        if (
            not stat_module.S_ISREG(after.st_mode)
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != before
        ):
            raise OSError(f"file changed while opening: {path}")
        handle = os.fdopen(fd, "rb")
        fd = -1
        return handle
    finally:
        if fd >= 0:
            os.close(fd)


def sha256_file(path: Path, max_bytes: int = MAX_HASH_BYTES) -> str:
    """Hash a regular file through a checked descriptor.

    ``lstat``/``open`` is otherwise a symlink-replacement race: the path can
    be changed after the first check and the hash would follow it outside the
    output tree.  ``O_NOFOLLOW`` closes that race on POSIX; descriptor and
    path identities are checked on both sides as an additional guard.
    """
    before_path = path.lstat()
    if not stat_module.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"output entry is not a regular file: {path}")
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        before_fd = os.fstat(fd)
        if not stat_module.S_ISREG(before_fd.st_mode):
            raise RuntimeError(f"output entry is not a regular file: {path}")
        if (before_fd.st_dev, before_fd.st_ino) != (before_path.st_dev, before_path.st_ino):
            raise RuntimeError(f"file changed before hashing it: {path}")
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(fd, "rb") as handle:
            fd = -1  # ownership transferred to the file object
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total += len(chunk)
                if total > MAX_HASH_BYTES:
                    raise RuntimeError(
                        f"file exceeds the {MAX_HASH_BYTES} byte hashing limit: {path}"
                    )
                digest.update(chunk)
            after_fd = os.fstat(handle.fileno())
        after_path = path.lstat()
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    if (
        (after_fd.st_dev, after_fd.st_ino, after_fd.st_size, after_fd.st_mtime_ns)
        != (before_fd.st_dev, before_fd.st_ino, before_fd.st_size, before_fd.st_mtime_ns)
        or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
        != (before_path.st_dev, before_path.st_ino, before_path.st_size, before_path.st_mtime_ns)
    ):
        raise RuntimeError(f"file changed while hashing it: {path}")
    return digest.hexdigest()


def directory_identity(path: Path) -> tuple[int, int]:
    """Return a stable identity for a directory without including its mtime.

    A benchmark is allowed to create files in its working directory.  Using
    the directory's size/mtime here would therefore reject an otherwise stable
    cwd after the first trial; device/inode identify replacement while allowing
    normal directory contents to change.
    """
    details = path.stat()
    if not stat_module.S_ISDIR(details.st_mode):
        raise NotADirectoryError(f"not a directory: {path}")
    return details.st_dev, details.st_ino


def snapshot_file_identity(path: str | None) -> tuple[int, int, int, int] | None:
    if path is None:
        return None
    try:
        return file_identity(Path(path))
    except (OSError, ValueError):
        return None


def atomic_write(path: Path, data: bytes, *, replace: bool = False) -> None:
    """Write and fsync a file without exposing a partial JSON/checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.tmp-"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # A pre-check followed by replace is a no-replace TOCTOU race.
            # Linking the private temporary inode into place is atomic and
            # fails if a child/attacker won the destination race.
            os.link(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_new_json(path: Path, value: Any) -> None:
    atomic_write(path, pretty_json(value), replace=False)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of silently keeping one value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def ensure_json_depth(raw: bytes) -> None:
    """Reject deeply nested JSON before the Python decoder can recurse."""
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue
        if byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise SpecError(f"spec JSON nesting exceeds the {MAX_JSON_DEPTH} level limit")
        elif byte in (ord("}"), ord("]")):
            depth = max(0, depth - 1)


def read_spec(path: Path) -> tuple[dict[str, Any], bytes, str, tuple[int, int, int, int] | None]:
    if str(path) == "-":
        raw = sys.stdin.buffer.read(MAX_SPEC_BYTES + 1)
        if len(raw) > MAX_SPEC_BYTES:
            raise SpecError(
                f"spec on stdin exceeds the {MAX_SPEC_BYTES} byte limit"
            )
        identity = None
    else:
        try:
            identity = file_identity(path)
        except OSError as error:
            raise SpecError(f"spec is not a regular file: {path} ({error})") from error
        if identity[2] > MAX_SPEC_BYTES:
            raise SpecError(
                f"spec file {path} is {identity[2]} bytes, exceeding the "
                f"{MAX_SPEC_BYTES} byte limit"
            )
        # Read through one descriptor with a hard cap. The initial path stat,
        # descriptor identity, fstat-after-read, and final path identity close
        # replacement/growth races without ever allocating an unbounded file.
        with open_regular_file(path) as handle:
            opened_stat = os.fstat(handle.fileno())
            opened_identity = (
                opened_stat.st_dev,
                opened_stat.st_ino,
                opened_stat.st_size,
                opened_stat.st_mtime_ns,
            )
            if opened_identity != identity:
                raise SpecError(f"spec file changed before reading it: {path}")
            raw = handle.read(MAX_SPEC_BYTES + 1)
            after_stat = os.fstat(handle.fileno())
            after_identity = (
                after_stat.st_dev,
                after_stat.st_ino,
                after_stat.st_size,
                after_stat.st_mtime_ns,
            )
        if len(raw) > MAX_SPEC_BYTES:
            raise SpecError(
                f"spec file {path} exceeds the {MAX_SPEC_BYTES} byte limit"
            )
        if after_identity != identity or file_identity(path) != identity:
            raise SpecError(f"spec file changed while reading it: {path}")
    ensure_json_depth(raw)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_json_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise SpecError(f"spec is not UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise SpecError("top-level spec must be an object")
    return value, raw, sha256_bytes(raw), identity


def has_surrogate(value: str) -> bool:
    return any(0xD800 <= ord(character) <= 0xDFFF for character in value)


def require_id(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > MAX_ID_CHARS
        or has_surrogate(value)
        or not ID_RE.fullmatch(value)
    ):
        raise SpecError(
            f"{field} must match [A-Za-z0-9][A-Za-z0-9_.-]* and be at most "
            f"{MAX_ID_CHARS} characters (got {value!r})"
        )
    return value


def require_string_list(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise SpecError(f"{field} must be a {'non-empty ' if nonempty else ''}list")
    if any(
        not isinstance(item, str)
        or "\x00" in item
        or has_surrogate(item)
        for item in value
    ):
        raise SpecError(f"{field} must contain strings without NUL or surrogate characters")
    return list(value)


def require_string_map(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise SpecError(f"{field} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or "\x00" in key
            or "=" in key
            or has_surrogate(key)
        ):
            raise SpecError(f"{field} has an invalid environment name")
        if not isinstance(item, str) or "\x00" in item or has_surrogate(item):
            raise SpecError(
                f"{field}.{key} must be a string without NUL or surrogate characters"
            )
        result[key] = item
    return result


def resolve_cwd(raw: Any, base_dir: Path, field: str) -> tuple[str | None, Path]:
    if raw is None:
        resolved = base_dir.resolve()
        return None, resolved
    if not isinstance(raw, str) or "\x00" in raw or has_surrogate(raw) or not raw:
        raise SpecError(
            f"{field} must be a non-empty path string without NUL/surrogate characters or null"
        )
    source = Path(raw)
    resolved = (source if source.is_absolute() else base_dir / source).resolve()
    if not resolved.is_dir():
        raise SpecError(f"{field} does not resolve to a directory: {resolved}")
    return raw, resolved


def finite_positive(value: Any, field: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{field} must be a positive number")
    number = value
    try:
        finite = math.isfinite(float(number))
    except OverflowError as error:
        raise SpecError(f"{field} is too large") from error
    if not finite or number <= 0:
        raise SpecError(f"{field} must be finite and positive")
    if integer and (not isinstance(number, int) or number < 1):
        raise SpecError(f"{field} must be a positive integer")
    return number


def nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpecError(f"{field} must be a non-negative integer")
    return value


def resolve_executable_path(argv: list[str], cwd: Path, env: dict[str, str]) -> Path | None:
    """Resolve the path that ``exec*`` will search for in ``cwd``.

    ``shutil.which`` normally searches relative to the harness process, while
    the child searches after changing to its requested cwd.  Normalize PATH
    entries first so a relative entry (including an empty entry) is resolved
    with the same base directory as the child.
    """
    candidate = Path(argv[0])
    try:
        if candidate.is_absolute() or "/" in argv[0] or "\\" in argv[0]:
            executable = candidate if candidate.is_absolute() else cwd / candidate
            executable = executable.resolve()
        else:
            path_entries = os.get_exec_path(env)
            absolute_entries: list[str] = []
            for entry in path_entries:
                search_dir = Path(entry) if entry else cwd
                if not search_dir.is_absolute():
                    search_dir = cwd / search_dir
                absolute_entries.append(str(search_dir.resolve()))
            executable_name = shutil.which(
                argv[0], path=os.pathsep.join(absolute_entries)
            )
            executable = Path(executable_name).resolve() if executable_name else None
    except (OSError, RuntimeError, ValueError):
        return None
    if executable is None or not executable.is_file():
        return None
    return executable


def resolve_executable(argv: list[str], cwd: Path, env: dict[str, str]) -> tuple[str | None, str | None]:
    """Resolve/hash only the executable, never model or other command inputs."""
    executable = resolve_executable_path(argv, cwd, env)
    if executable is None:
        return None, None
    try:
        return str(executable), sha256_file(executable)
    except (OSError, RuntimeError):
        return str(executable), None


def executable_snapshot(
    argv: list[str], cwd: Path, env: dict[str, str]
) -> dict[str, Any]:
    """Capture executable path, content hash, and inode identity together."""
    executable, executable_sha256 = resolve_executable(argv, cwd, env)
    return {
        "executable": executable,
        "executable_sha256": executable_sha256,
        "executable_identity": snapshot_file_identity(executable),
    }


def revalidate_trial_identity(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Check cwd and executable snapshots immediately before spawning a trial.

    This closes the normal validation-to-trial gap: a changed command or
    replaced cwd is recorded as a failed trial and is never launched.  The
    final filesystem lookup/exec still has an unavoidable tiny race on APIs
    without fd-based cwd/exec support, so the result records both snapshots
    and the policy remains fail-closed on every detectable mismatch.
    """
    try:
        actual_cwd_identity = directory_identity(cwd)
        cwd_error = None
    except (OSError, ValueError) as error:
        actual_cwd_identity = None
        cwd_error = f"{type(error).__name__}: {error}"

    try:
        actual = executable_snapshot(argv, cwd, env)
        executable_error = None
    except (OSError, RuntimeError, ValueError) as error:
        actual = {
            "executable": None,
            "executable_sha256": None,
            "executable_identity": None,
        }
        executable_error = f"{type(error).__name__}: {error}"

    expected_cwd_identity = expected.get("cwd_identity")
    expected_executable = expected.get("executable")
    expected_executable_sha256 = expected.get("executable_sha256")
    expected_executable_identity = expected.get("executable_identity")
    errors: list[str] = []
    if cwd_error is not None:
        errors.append(f"cwd cannot be inspected ({cwd_error})")
    elif actual_cwd_identity != expected_cwd_identity:
        errors.append(
            "cwd identity changed "
            f"(expected {expected_cwd_identity!r}, got {actual_cwd_identity!r})"
        )
    if executable_error is not None:
        errors.append(f"executable cannot be inspected ({executable_error})")
    if actual["executable"] != expected_executable:
        errors.append(
            "executable path changed "
            f"(expected {expected_executable!r}, got {actual['executable']!r})"
        )
    if actual["executable_sha256"] != expected_executable_sha256:
        errors.append("executable content hash changed or is unavailable")
    if actual["executable_identity"] != expected_executable_identity:
        errors.append(
            "executable file identity changed "
            f"(expected {expected_executable_identity!r}, "
            f"got {actual['executable_identity']!r})"
        )
    # Never fall back to a fresh PATH lookup when preflight could not resolve
    # the executable: a binary may appear between validation and exec.
    if expected_executable is None:
        errors.append("preflight executable was unavailable; refusing an unverified PATH lookup")
    # A known executable must always have a hash.  Otherwise the path could be
    # stable while its bytes remain unverified, which is not safe to benchmark.
    elif expected_executable_sha256 is None:
        errors.append("preflight executable hash was unavailable")
    if actual["executable"] is not None and actual["executable_sha256"] is None:
        errors.append("trial executable hash was unavailable")

    return {
        "ok": not errors,
        "error": "; ".join(errors) if errors else None,
        "cwd_identity": actual_cwd_identity,
        **actual,
    }


def resource_value(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value
    return None


def usage_snapshot() -> Any:
    if resource is None:
        return None
    try:
        return resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, OSError):
        return None


def usage_delta(before: Any, after: Any) -> dict[str, int | float | None]:
    if before is None or after is None:
        return {
            "user_cpu_s": None,
            "system_cpu_s": None,
            "minor_page_faults": None,
            "major_page_faults": None,
            "voluntary_context_switches": None,
            "involuntary_context_switches": None,
            "max_rss_bytes_fallback": None,
        }

    def difference(field: str) -> int | float | None:
        left = getattr(before, field, None)
        right = getattr(after, field, None)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        return max(0, right - left)

    # ru_maxrss is KiB on Linux, bytes on macOS/BSD.  It is a fallback only;
    # Linux's /proc VmHWM sampler below is preferred and per-process.
    maxrss = getattr(after, "ru_maxrss", None)
    maxrss_bytes = None
    if isinstance(maxrss, (int, float)) and maxrss >= 0:
        maxrss_bytes = int(maxrss * (1024 if sys.platform.startswith("linux") else 1))
    return {
        "user_cpu_s": resource_value(difference("ru_utime")),
        "system_cpu_s": resource_value(difference("ru_stime")),
        "minor_page_faults": difference("ru_minflt"),
        "major_page_faults": difference("ru_majflt"),
        "voluntary_context_switches": difference("ru_nvcsw"),
        "involuntary_context_switches": difference("ru_nivcsw"),
        "max_rss_bytes_fallback": maxrss_bytes,
    }


def proc_peak_rss_bytes(pid: int) -> int | None:
    """Read Linux's per-process high-water mark when available."""
    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text(encoding="ascii", errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(("VmHWM:", "VmRSS:")):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    values[fields[0][:-1]] = int(fields[1]) * 1024
                except ValueError:
                    pass
    # VmHWM is a high-water mark and is preferable to one instantaneous RSS.
    return values.get("VmHWM") or values.get("VmRSS")


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=2.0)


def terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the process group even if the direct child already exited."""
    if os.name != "posix":
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.perf_counter() + 1.0
    while proc.poll() is None and time.perf_counter() < deadline:
        time.sleep(0.01)
    # A detached descendant can keep the group alive after the direct child
    # exits; SIGKILL closes that leak and unblocks pipe readers.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def output_bytes(path: Path, max_bytes: int) -> tuple[int, str]:
    """Hash a bounded output file through an identity-checked descriptor."""
    before_path = path.lstat()
    if not stat_module.S_ISREG(before_path.st_mode):
        raise RuntimeError(f"captured output is not a regular file: {path}")
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        before_fd = os.fstat(fd)
        if (
            not stat_module.S_ISREG(before_fd.st_mode)
            or (before_fd.st_dev, before_fd.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise RuntimeError(f"captured output changed before hashing: {path}")
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                if size > max_bytes:
                    raise RuntimeError(
                        f"captured output exceeds the {max_bytes} byte limit: {path}"
                    )
                digest.update(chunk)
            after_fd = os.fstat(handle.fileno())
        after_path = path.lstat()
    except BaseException:
        if fd >= 0:
            os.close(fd)
        raise
    if (
        (after_fd.st_dev, after_fd.st_ino, after_fd.st_size, after_fd.st_mtime_ns)
        != (before_fd.st_dev, before_fd.st_ino, before_fd.st_size, before_fd.st_mtime_ns)
        or (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
        != (before_path.st_dev, before_path.st_ino, before_path.st_size, before_path.st_mtime_ns)
    ):
        raise RuntimeError(f"captured output changed while hashing: {path}")
    return size, digest.hexdigest()


class _PipeCaptureState:
    """Shared byte budget for the stdout/stderr reader threads."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.total = 0
        self.exceeded = False
        self.closing = False
        self.error: str | None = None


def _capture_pipe(stream: Any, output: Any, state: _PipeCaptureState) -> None:
    """Drain one child pipe while enforcing the combined output budget."""
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            with state.lock:
                remaining = state.max_bytes - state.total
                if remaining <= 0:
                    state.exceeded = True
                    return
                if len(chunk) > remaining:
                    output.write(chunk[:remaining])
                    state.total += remaining
                    state.exceeded = True
                    return
                output.write(chunk)
                state.total += len(chunk)
    except Exception as error:
        # Closing a pipe after timeout/limit termination is expected. Other
        # failures (for example, a full output filesystem) invalidate the trial.
        with state.lock:
            if not state.closing:
                state.error = f"{type(error).__name__}: {error}"


def _close_process_pipes(process: subprocess.Popen[bytes], state: _PipeCaptureState) -> None:
    with state.lock:
        state.closing = True
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass


def run_trial(
    *,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    max_output_bytes: int,
    stdout_path: Path,
    stderr_path: Path,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=False)
    # Binary files preserve output exactly, including invalid UTF-8 and no final
    # newline for complete trials. Output-limit failures retain bounded
    # per-stream prefixes under the combined cap.
    # A pair of reader threads drains stdout/stderr concurrently while
    # enforcing one combined byte budget before data reaches disk.
    start = utc_now()
    monotonic_start = time.perf_counter()
    before_usage = usage_snapshot()
    process: subprocess.Popen[bytes] | None = None
    process_started_monotonic: float | None = None
    process_finished_monotonic: float | None = None
    launch_error: str | None = None
    identity_error: str | None = None
    identity_result: dict[str, Any] | None = None
    capture_error: str | None = None
    capture_incomplete = False
    timed_out = False
    output_limit_exceeded = False
    peak_rss = 0
    peak_rss_source: str | None = None
    returncode: int | None = None
    state: _PipeCaptureState | None = None
    threads: list[threading.Thread] = []
    capture_thread_leaked = False

    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        if expected_identity is not None:
            identity_result = revalidate_trial_identity(
                argv=argv, cwd=cwd, env=env, expected=expected_identity
            )
            if not identity_result["ok"]:
                identity_error = identity_result["error"]
        try:
            kwargs: dict[str, Any] = {
                "args": argv,
                "cwd": str(cwd),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "bufsize": 0,
                "close_fds": True,
            }
            if identity_result is not None and identity_result["executable"] is not None:
                # Keep argv[0] unchanged for the child while avoiding a second
                # PATH lookup after the identity check.
                kwargs["executable"] = identity_result["executable"]
            if os.name == "posix":
                kwargs["start_new_session"] = True
            if identity_error is None:
                process = subprocess.Popen(**kwargs)
                process_started_monotonic = time.perf_counter()
        except (OSError, ValueError) as error:
            launch_error = f"{type(error).__name__}: {error}"
        if process is not None:
            state = _PipeCaptureState(max_output_bytes)
            try:
                for stream, output in ((process.stdout, stdout), (process.stderr, stderr)):
                    if stream is None:  # defensive; Popen(PIPE) guarantees both
                        continue
                    thread = threading.Thread(
                        target=_capture_pipe,
                        args=(stream, output, state),
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)

                while True:
                    sample = proc_peak_rss_bytes(process.pid)
                    if sample is not None and sample > peak_rss:
                        peak_rss = sample
                        peak_rss_source = "proc-vmhwm"
                    with state.lock:
                        exceeded = state.exceeded
                        capture_error = state.error
                    if exceeded:
                        output_limit_exceeded = True
                        terminate_process(process)
                        break
                    if capture_error is not None:
                        terminate_process(process)
                        break
                    returncode = process.poll()
                    if returncode is not None:
                        process_finished_monotonic = time.perf_counter()
                        break
                    if (
                        process_started_monotonic is not None
                        and time.perf_counter() - process_started_monotonic >= timeout_s
                    ):
                        timed_out = True
                        terminate_process(process)
                        break
                    time.sleep(POLL_SECONDS)

                if returncode is None:
                    # Defensive wait after a race with timeout/termination.
                    try:
                        returncode = process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        terminate_process(process)
                        returncode = process.returncode
                    if process_finished_monotonic is None:
                        process_finished_monotonic = time.perf_counter()
            finally:
                # Always tear down the process group and pipes, including when
                # Ctrl-C or an unexpected monitor error unwinds this function.
                # On normal exit, allow already-written bytes to drain first;
                # descendants retaining a pipe are bounded by the join below.
                if process.poll() is None:
                    terminate_process(process)
                if process_finished_monotonic is None and process.poll() is not None:
                    process_finished_monotonic = time.perf_counter()
                if timed_out or output_limit_exceeded or capture_error is not None:
                    # The direct child may have exited while a descendant still
                    # owns a pipe. Kill the group even when poll() is non-None.
                    terminate_process_group(process)
                    _close_process_pipes(process, state)
                else:
                    for thread in threads:
                        thread.join(timeout=1.0)
                    if any(thread.is_alive() for thread in threads):
                        capture_incomplete = True
                        terminate_process_group(process)
                        _close_process_pipes(process, state)
                for thread in threads:
                    thread.join(timeout=1.0)
                if any(thread.is_alive() for thread in threads):
                    capture_incomplete = True
                    capture_thread_leaked = True
                    _close_process_pipes(process, state)
                # Cleanup is unconditional on POSIX: a successful direct child
                # may have forked a descendant that closed its stdio handles,
                # so pipe closure alone is not evidence that the group is gone.
                terminate_process_group(process)
                with state.lock:
                    output_limit_exceeded = output_limit_exceeded or state.exceeded
                    capture_error = capture_error or state.error
                if capture_incomplete and capture_error is None:
                    capture_error = "output pipe did not close after the child exited"
                sample = proc_peak_rss_bytes(process.pid)
                if sample is not None and sample > peak_rss:
                    peak_rss = sample
                    peak_rss_source = "proc-vmhwm"

    elapsed_s = time.perf_counter() - monotonic_start
    finished_at = utc_now()
    after_usage = usage_snapshot()
    usage = usage_delta(before_usage, after_usage)
    if not peak_rss:
        fallback = usage.get("max_rss_bytes_fallback")
        if isinstance(fallback, int):
            peak_rss = fallback
            peak_rss_source = "rusage-fallback"
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    stdout_bytes, stdout_sha256 = 0, empty_sha256
    stderr_bytes, stderr_sha256 = 0, empty_sha256
    for stream_name, stream_path in (("stdout", stdout_path), ("stderr", stderr_path)):
        try:
            stream_bytes, stream_sha256 = output_bytes(stream_path, max_output_bytes)
        except (OSError, RuntimeError, ValueError) as error:
            capture_error = capture_error or f"{stream_name} capture validation failed: {error}"
            continue
        if stream_name == "stdout":
            stdout_bytes, stdout_sha256 = stream_bytes, stream_sha256
        else:
            stderr_bytes, stderr_sha256 = stream_bytes, stream_sha256
    result: dict[str, Any] = {
        "started_at": start,
        "finished_at": finished_at,
        # elapsed_s includes bounded capture/drain time. process_elapsed_s is
        # sampled at child exit and is the fairer metric for runtime comparison.
        "elapsed_s": elapsed_s,
        "process_elapsed_s": (
            max(0.0, process_finished_monotonic - process_started_monotonic)
            if process_started_monotonic is not None
            and process_finished_monotonic is not None
            else None
        ),
        "identity_revalidated": expected_identity is not None,
        "identity_error": identity_error,
        "cwd_identity": (
            list(identity_result["cwd_identity"])
            if identity_result is not None
            and identity_result["cwd_identity"] is not None
            else None
        ),
        "executable": (
            identity_result["executable"] if identity_result is not None else None
        ),
        "executable_sha256": (
            identity_result["executable_sha256"]
            if identity_result is not None
            else None
        ),
        "executable_identity": (
            list(identity_result["executable_identity"])
            if identity_result is not None
            and identity_result["executable_identity"] is not None
            else None
        ),
        "expected_cwd_identity": (
            list(expected_identity["cwd_identity"])
            if expected_identity is not None
            and expected_identity.get("cwd_identity") is not None
            else None
        ),
        "expected_executable": (
            expected_identity.get("executable")
            if expected_identity is not None
            else None
        ),
        "expected_executable_sha256": (
            expected_identity.get("executable_sha256")
            if expected_identity is not None
            else None
        ),
        "expected_executable_identity": (
            list(expected_identity["executable_identity"])
            if expected_identity is not None
            and expected_identity.get("executable_identity") is not None
            else None
        ),
        "returncode": returncode,
        "timed_out": timed_out,
        "output_limit_exceeded": output_limit_exceeded,
        "launch_error": launch_error,
        "capture_error": capture_error,
        "capture_incomplete": capture_incomplete,
        "capture_thread_leaked": capture_thread_leaked,
        "stdout": {
            "path": stdout_path.as_posix(),
            "bytes": stdout_bytes,
            "sha256": stdout_sha256,
        },
        "stderr": {
            "path": stderr_path.as_posix(),
            "bytes": stderr_bytes,
            "sha256": stderr_sha256,
        },
        "resource": {
            "user_cpu_s": usage["user_cpu_s"],
            "system_cpu_s": usage["system_cpu_s"],
            "minor_page_faults": usage["minor_page_faults"],
            "major_page_faults": usage["major_page_faults"],
            "voluntary_context_switches": usage["voluntary_context_switches"],
            "involuntary_context_switches": usage["involuntary_context_switches"],
            "peak_rss_bytes": peak_rss or None,
            "peak_rss_source": peak_rss_source,
        },
    }
    if identity_error:
        result["status"] = "identity-mismatch"
    elif launch_error:
        result["status"] = "launch-failed"
    elif capture_error:
        result["status"] = "capture-failed"
    elif timed_out:
        result["status"] = "timed-out"
    elif output_limit_exceeded:
        result["status"] = "output-limit-exceeded"
    elif returncode == 0:
        result["status"] = "ok"
    else:
        result["status"] = "exit-failed"
    return result


def validate_spec(
    raw: dict[str, Any], *, spec_base: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if raw.get("schema") != SCHEMA:
        raise SpecError(f"spec.schema must be {SCHEMA!r}")
    benchmark_id = require_id(raw.get("id"), "spec.id")
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise SpecError("spec.description must be a string")
    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        raise SpecError("spec.inputs must be an object")
    runtimes_raw = raw.get("runtimes")
    if not isinstance(runtimes_raw, list) or not runtimes_raw:
        raise SpecError("spec.runtimes must be a non-empty list")
    if len(runtimes_raw) > MAX_RUNTIMES:
        raise SpecError(
            f"spec.runtimes has {len(runtimes_raw)} entries; limit is {MAX_RUNTIMES}"
        )
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise SpecError("spec.cases must be a non-empty list")
    if len(cases_raw) > MAX_CASES:
        raise SpecError(
            f"spec.cases has {len(cases_raw)} entries; limit is {MAX_CASES}"
        )

    runtimes: list[dict[str, Any]] = []
    runtime_ids: set[str] = set()
    for index, value in enumerate(runtimes_raw):
        field = f"spec.runtimes[{index}]"
        if not isinstance(value, dict):
            raise SpecError(f"{field} must be an object")
        runtime_id = require_id(value.get("id"), f"{field}.id")
        if runtime_id in runtime_ids:
            raise SpecError(f"duplicate runtime id: {runtime_id}")
        runtime_ids.add(runtime_id)
        argv = require_string_list(value.get("command"), f"{field}.command", nonempty=True)
        inherit_env = value.get("inherit_env", False)
        if not isinstance(inherit_env, bool):
            raise SpecError(f"{field}.inherit_env must be boolean")
        env_overrides = require_string_map(value.get("env", {}), f"{field}.env")
        cwd_raw, cwd = resolve_cwd(value.get("cwd"), spec_base, f"{field}.cwd")
        try:
            cwd_identity = directory_identity(cwd)
        except (OSError, ValueError) as error:
            raise SpecError(f"{field}.cwd cannot be inspected: {error}") from error
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SpecError(f"{field}.metadata must be an object")
        env = (dict(os.environ) if inherit_env else {})
        env.update(env_overrides)
        # Executable hashing is deferred until all scalar/matrix caps pass.
        runtimes.append(
            {
                "id": runtime_id,
                "command": argv,
                "cwd_spec": cwd_raw,
                "cwd": str(cwd),
                "inherit_env": inherit_env,
                "env": env_overrides,
                "metadata": metadata,
                "_resolved_env": env,
                "_executable": None,
                "_executable_sha256": None,
                "_executable_identity": None,
                "_cwd_identity": cwd_identity,
            }
        )

    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for index, value in enumerate(cases_raw):
        field = f"spec.cases[{index}]"
        if not isinstance(value, dict):
            raise SpecError(f"{field} must be an object")
        case_id = require_id(value.get("id"), f"{field}.id")
        if case_id in case_ids:
            raise SpecError(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        args = require_string_list(value.get("args", []), f"{field}.args")
        warmups = nonnegative_integer(value.get("warmups", 1), f"{field}.warmups")
        if warmups > MAX_WARMUPS:
            raise SpecError(
                f"{field}.warmups is {warmups}; limit is {MAX_WARMUPS}"
            )
        repetitions = finite_positive(value.get("repetitions", 3), f"{field}.repetitions", integer=True)
        if repetitions > MAX_REPETITIONS:
            raise SpecError(
                f"{field}.repetitions is {repetitions}; limit is {MAX_REPETITIONS}"
            )
        timeout_s = finite_positive(value.get("timeout_s", 600.0), f"{field}.timeout_s")
        if timeout_s > MAX_TIMEOUT_SECONDS:
            raise SpecError(
                f"{field}.timeout_s is {timeout_s}; limit is {MAX_TIMEOUT_SECONDS} seconds"
            )
        max_output_bytes = finite_positive(
            value.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES),
            f"{field}.max_output_bytes",
            integer=True,
        )
        if max_output_bytes > MAX_OUTPUT_BYTES:
            raise SpecError(
                f"{field}.max_output_bytes is {max_output_bytes}; limit is {MAX_OUTPUT_BYTES}"
            )
        case_env = require_string_map(value.get("env", {}), f"{field}.env")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise SpecError(f"{field}.metadata must be an object")
        cases.append(
            {
                "id": case_id,
                "args": args,
                "warmups": warmups,
                "repetitions": repetitions,
                "timeout_s": float(timeout_s),
                "max_output_bytes": max_output_bytes,
                "env": case_env,
                "metadata": metadata,
            }
        )

    total_trials = sum(
        len(runtimes) * (case["warmups"] + case["repetitions"])
        for case in cases
    )
    if total_trials > MAX_TOTAL_TRIALS:
        raise SpecError(
            f"benchmark matrix has {total_trials} trials; limit is {MAX_TOTAL_TRIALS}"
        )
    total_timeout_seconds = sum(
        len(runtimes) * (case["warmups"] + case["repetitions"]) * case["timeout_s"]
        for case in cases
    )
    if total_timeout_seconds > MAX_TOTAL_TIMEOUT_SECONDS:
        raise SpecError(
            f"benchmark matrix declares {total_timeout_seconds:g} timeout seconds; "
            f"limit is {MAX_TOTAL_TIMEOUT_SECONDS}"
        )
    total_output_bytes = sum(
        len(runtimes)
        * (case["warmups"] + case["repetitions"])
        * case["max_output_bytes"]
        for case in cases
    )
    if total_output_bytes > MAX_TOTAL_OUTPUT_BYTES:
        raise SpecError(
            f"benchmark matrix permits {total_output_bytes} output bytes; "
            f"limit is {MAX_TOTAL_OUTPUT_BYTES}"
        )

    # Executable lookup depends on the effective environment: a case may
    # override PATH. Capture one immutable snapshot for every matrix cell only
    # after all scalar/matrix caps pass. Cache identical PATH lookups so a
    # large-but-allowed matrix does not repeatedly hash the same executable.
    snapshot_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for runtime in runtimes:
        base_snapshot = executable_snapshot(
            runtime["command"], Path(runtime["cwd"]), runtime["_resolved_env"]
        )
        runtime["_executable"] = base_snapshot["executable"]
        runtime["_executable_sha256"] = base_snapshot["executable_sha256"]
        runtime["_executable_identity"] = base_snapshot["executable_identity"]
    for case in cases:
        identities: dict[str, dict[str, Any]] = {}
        for runtime in runtimes:
            effective_env = dict(runtime["_resolved_env"])
            effective_env.update(case["env"])
            cache_key = (
                tuple(runtime["command"]),
                runtime["cwd"],
                effective_env.get("PATH"),
                effective_env.get("PATHEXT"),
            )
            snapshot = snapshot_cache.get(cache_key)
            if snapshot is None:
                snapshot = executable_snapshot(
                    runtime["command"], Path(runtime["cwd"]), effective_env
                )
                snapshot_cache[cache_key] = snapshot
            identities[runtime["id"]] = {
                "cwd_identity": runtime["_cwd_identity"],
                **snapshot,
            }
        case["_runtime_identities"] = identities

    # Keep the normalized copy free of private runtime data.  The source spec
    # remains in the manifest, while this matrix is the exact execution plan.
    return {
        "schema": SCHEMA,
        "id": benchmark_id,
        "description": description,
        "inputs": inputs,
        "runtimes": [
            {
                key: value
                for key, value in runtime.items()
                if not key.startswith("_") and key != "_resolved_env"
            }
            for runtime in runtimes
        ],
        "cases": [
            {
                key: value
                for key, value in case.items()
                if not key.startswith("_")
            }
            for case in cases
        ],
    }, runtimes, cases


def environment_digest(env: dict[str, str]) -> str:
    return sha256_bytes(canonical_json(env))


def repository_facts() -> dict[str, Any]:
    """Best-effort source identity; the runner hash remains authoritative."""
    root = SCRIPT_PATH.parent.parent
    facts: dict[str, Any] = {"root": str(root), "commit": None, "working_tree_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=5,
        )
        if commit.returncode == 0 and commit.stdout.strip():
            facts["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=5,
        )
        if status.returncode == 0:
            facts["working_tree_dirty"] = bool(status.stdout)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return facts


def host_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "page_size": None,
    }
    try:
        facts["page_size"] = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pass
    if hasattr(os, "uname"):
        uname = os.uname()
        facts["uname"] = {
            "sysname": uname.sysname,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        }
    return facts


def build_manifest(
    *,
    spec: dict[str, Any],
    raw_spec_sha256: str,
    spec_path: Path,
    spec_identity: tuple[int, int, int, int] | None,
    normalized: dict[str, Any],
    runtimes: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    script_sha256 = sha256_file(SCRIPT_PATH)
    matrix: list[dict[str, Any]] = []
    for case in cases:
        for runtime in runtimes:
            env = dict(runtime["_resolved_env"])
            env.update(case["env"])
            expected = case.get("_runtime_identities", {}).get(runtime["id"])
            if expected is None:
                # Keep this helper usable by callers constructing normalized
                # matrices directly rather than through validate_spec().
                expected = {
                    "cwd_identity": runtime["_cwd_identity"],
                    "executable": runtime["_executable"],
                    "executable_sha256": runtime["_executable_sha256"],
                    "executable_identity": runtime.get("_executable_identity"),
                }
            matrix.append(
                {
                    "runtime_id": runtime["id"],
                    "case_id": case["id"],
                    "command": runtime["command"] + case["args"],
                    "cwd": runtime["cwd"],
                    "cwd_spec": runtime["cwd_spec"],
                    "cwd_identity": list(expected["cwd_identity"]),
                    "runtime_env_overrides": runtime["env"],
                    "case_env_overrides": case["env"],
                    "inherit_env": runtime["inherit_env"],
                    # Values are not copied into the manifest because a host
                    # environment can contain credentials.  This hash exposes
                    # drift without disclosing them.
                    "effective_environment_names": sorted(env),
                    "effective_environment_sha256": environment_digest(env),
                    "executable": expected["executable"],
                    "executable_sha256": expected["executable_sha256"],
                    "executable_identity": (
                        list(expected["executable_identity"])
                        if expected["executable_identity"] is not None
                        else None
                    ),
                    "timeout_s": case["timeout_s"],
                    "max_output_bytes": case["max_output_bytes"],
                    "warmups": case["warmups"],
                    "repetitions": case["repetitions"],
                }
            )
    manifest = {
        "schema": SCHEMA,
        "manifest_version": 1,
        "created_at": utc_now(),
        "benchmark_id": normalized["id"],
        "description": normalized["description"],
        "output_directory": str(output.resolve()),
        "spec": spec,
        "spec_canonical_sha256": sha256_bytes(canonical_json(spec)),
        "spec_input_sha256": raw_spec_sha256,
        "spec_source": "<stdin>" if str(spec_path) == "-" else str(spec_path.resolve()),
        "spec_source_identity": list(spec_identity) if spec_identity is not None else None,
        "runner": {
            "path": str(SCRIPT_PATH),
            "sha256": script_sha256,
            "python": sys.version,
        },
        "repository": repository_facts(),
        "host": host_facts(),
        "execution": {
            "command_shell": False,
            "order": "case list, warmups then measured repetitions, runtime list within each step",
            "resource_sampler": "Linux /proc VmHWM with getrusage fallback",
            "timing": {
                "wall_seconds": "process_elapsed_s: Popen return to child exit",
                "capture_wall_seconds": "elapsed_s: launch through bounded output drain",
                "timeout": "elapsed_s includes launch and capture; process is terminated at the bound",
            },
            "stdout_stderr": {
                "format": "binary files; no decoding or newline normalization",
                "complete_trial": "exact bytes",
                "output_limit_failure": "captured per-stream prefixes under one combined stdout+stderr byte cap",
                "hash": "SHA-256 of captured bytes",
            },
            "identity_policy": (
                "cwd device/inode and executable resolved path, device/inode, and SHA-256 "
                "are snapshotted before the manifest; each trial revalidates immediately "
                "before Popen and skips mismatches (the residual check-to-exec race is recorded)"
            ),
            "failure_policy": "continue matrix and record every failure",
        },
        "matrix": matrix,
        "artifact_layout": {
            "manifest": "manifest.json",
            "manifest_sha256": "manifest.sha256",
            "trial_records": "trials/<runtime>/<case>/<warmup|run>-NNN/trial.json",
            "stdout": "trials/<runtime>/<case>/<warmup|run>-NNN/stdout.bin",
            "stderr": "trials/<runtime>/<case>/<warmup|run>-NNN/stderr.bin",
            "results": "results.json",
            "summary": "summary.json",
            "checksums": "checksums.sha256",
        },
        "immutability": {
            "manifest_written_before_trials": True,
            "manifest_not_rewritten": True,
            "verify_with": "sha256sum -c manifest.sha256",
        },
    }
    return manifest


def relative_output(path: Path, output: Path) -> str:
    return path.relative_to(output).as_posix()


def trial_path(output: Path, runtime_id: str, case_id: str, phase: str, index: int) -> Path:
    return output / "trials" / runtime_id / case_id / f"{phase}-{index:03d}"


def stats(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
    }


def summarize_records(
    records: list[dict[str, Any]], runtime_ids: list[str], case_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        if not record["warmup"]:
            by_key.setdefault((record["runtime_id"], record["case_id"]), []).append(record)
    for runtime_id in runtime_ids:
        for case_id in case_ids:
            rows = by_key.get((runtime_id, case_id), [])
            successful = [row for row in rows if row["status"] == "ok"]
            stdout_hashes = sorted({row["stdout"]["sha256"] for row in rows})
            stderr_hashes = sorted({row["stderr"]["sha256"] for row in rows})
            successful_stdout_hashes = sorted({row["stdout"]["sha256"] for row in successful})
            successful_stderr_hashes = sorted({row["stderr"]["sha256"] for row in successful})
            process_elapsed = [
                row["process_elapsed_s"]
                for row in successful
                if row.get("process_elapsed_s") is not None
            ]
            capture_elapsed = [row["elapsed_s"] for row in successful]
            rss = [
                row["resource"]["peak_rss_bytes"]
                for row in successful
                if row["resource"]["peak_rss_bytes"] is not None
            ]
            user = [row["resource"]["user_cpu_s"] for row in successful if row["resource"]["user_cpu_s"] is not None]
            system = [
                row["resource"]["system_cpu_s"]
                for row in successful
                if row["resource"]["system_cpu_s"] is not None
            ]
            minor_faults = [
                row["resource"]["minor_page_faults"]
                for row in successful
                if row["resource"]["minor_page_faults"] is not None
            ]
            major_faults = [
                row["resource"]["major_page_faults"]
                for row in successful
                if row["resource"]["major_page_faults"] is not None
            ]
            voluntary_switches = [
                row["resource"]["voluntary_context_switches"]
                for row in successful
                if row["resource"]["voluntary_context_switches"] is not None
            ]
            involuntary_switches = [
                row["resource"]["involuntary_context_switches"]
                for row in successful
                if row["resource"]["involuntary_context_switches"] is not None
            ]
            returncodes = list({row["returncode"] for row in rows})
            returncodes.sort(key=lambda code: -1 if code is None else code)
            summaries.append(
                {
                    "runtime_id": runtime_id,
                    "case_id": case_id,
                    "measured_repetitions": len(rows),
                    "successful_repetitions": len(successful),
                    "failed_repetitions": len(rows) - len(successful),
                    "status": "ok" if rows and len(successful) == len(rows) else ("failed" if rows else "missing"),
                    # wall_seconds is the child-lifetime metric used by
                    # pairwise ratios; capture_wall_seconds includes bounded
                    # reader/drain overhead and is retained for auditability.
                    "wall_seconds": stats(process_elapsed),
                    "capture_wall_seconds": stats(capture_elapsed),
                    "peak_rss_bytes": stats(rss),
                    "user_cpu_seconds": stats(user),
                    "system_cpu_seconds": stats(system),
                    "minor_page_faults": stats(minor_faults),
                    "major_page_faults": stats(major_faults),
                    "voluntary_context_switches": stats(voluntary_switches),
                    "involuntary_context_switches": stats(involuntary_switches),
                    "returncodes": returncodes,
                    "stdout_sha256s": stdout_hashes,
                    "stderr_sha256s": stderr_hashes,
                    "successful_stdout_sha256s": successful_stdout_hashes,
                    "successful_stderr_sha256s": successful_stderr_hashes,
                    "stdout_deterministic": len(successful_stdout_hashes) == 1 if successful else None,
                    "stderr_deterministic": len(successful_stderr_hashes) == 1 if successful else None,
                }
            )
    comparisons: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_summaries = [row for row in summaries if row["case_id"] == case_id]
        for left, right in itertools.combinations(case_summaries, 2):
            def ratio_for(metric: str) -> float | None:
                left_value = left[metric]["median"]
                right_value = right[metric]["median"]
                if (
                    isinstance(left_value, (int, float))
                    and isinstance(right_value, (int, float))
                    and right_value > 0
                ):
                    return left_value / right_value
                return None

            left_stdout = left["successful_stdout_sha256s"]
            right_stdout = right["successful_stdout_sha256s"]
            left_stderr = left["successful_stderr_sha256s"]
            right_stderr = right["successful_stderr_sha256s"]
            comparisons.append(
                {
                    "case_id": case_id,
                    "runtime_a": left["runtime_id"],
                    "runtime_b": right["runtime_id"],
                    "both_complete": left["status"] == "ok" and right["status"] == "ok",
                    "wall_median_ratio_a_over_b": ratio_for("wall_seconds"),
                    "capture_wall_median_ratio_a_over_b": ratio_for("capture_wall_seconds"),
                    "peak_rss_median_ratio_a_over_b": ratio_for("peak_rss_bytes"),
                    "user_cpu_median_ratio_a_over_b": ratio_for("user_cpu_seconds"),
                    "system_cpu_median_ratio_a_over_b": ratio_for("system_cpu_seconds"),
                    "stdout_hash_equal": (
                        None
                        if left["stdout_deterministic"] is None
                        or right["stdout_deterministic"] is None
                        else bool(
                            left["stdout_deterministic"]
                            and right["stdout_deterministic"]
                            and left_stdout == right_stdout
                        )
                    ),
                    "stderr_hash_equal": (
                        None
                        if left["stderr_deterministic"] is None
                        or right["stderr_deterministic"] is None
                        else bool(
                            left["stderr_deterministic"]
                            and right["stderr_deterministic"]
                            and left_stderr == right_stderr
                        )
                    ),
                    "exit_outcome": {
                        "a": left["status"],
                        "b": right["status"],
                    },
                    "interpretation": (
                        "descriptive timing/output comparison only; neither runtime "
                        "is a correctness oracle"
                    ),
                }
            )
    return summaries, comparisons


def collect_files(output: Path) -> list[Path]:
    files: list[Path] = []
    entries = 0
    checksum_path = output / "checksums.sha256"
    for path in output.rglob("*"):
        if path == checksum_path:
            continue
        entries += 1
        if entries > MAX_OUTPUT_ENTRIES:
            raise RuntimeError(
                f"output tree contains more than the {MAX_OUTPUT_ENTRIES} entry limit"
            )
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise RuntimeError(f"cannot inspect output entry {path}: {error}") from error
        try:
            relative = path.relative_to(output)
            depth = len(relative.parts)
        except ValueError as error:
            raise RuntimeError(f"output entry escaped its root: {path}") from error
        if any(
            any(
                ord(character) < 0x20
                or 0xD800 <= ord(character) <= 0xDFFF
                or character == "\\"
                for character in part
            )
            for part in relative.parts
        ):
            raise RuntimeError(
                f"output entry contains a control character or backslash: {path}"
            )
        if depth > MAX_OUTPUT_PATH_DEPTH:
            raise RuntimeError(
                f"output path depth exceeds the {MAX_OUTPUT_PATH_DEPTH} level limit: {path}"
            )
        if stat_module.S_ISDIR(mode):
            continue
        if not stat_module.S_ISREG(mode):
            raise RuntimeError(f"output contains a non-regular or symlink entry: {path}")
        files.append(path)
        if len(files) > MAX_OUTPUT_FILES:
            raise RuntimeError(
                f"output tree contains more than the {MAX_OUTPUT_FILES} file limit"
            )
    return sorted(files)


def check_output_tree_size(output: Path, max_bytes: int) -> int:
    """Bound files a runtime may create in the harness output tree."""
    total = 0
    for path in collect_files(output):
        try:
            size = path.lstat().st_size
        except OSError as error:
            raise RuntimeError(f"cannot inspect output entry {path}: {error}") from error
        total += size
        if total > max_bytes:
            raise RuntimeError(
                f"output tree exceeds the {max_bytes} byte limit after runtime writes"
            )
    return total


def write_checksums(output: Path) -> None:
    lines: list[str] = []
    for path in collect_files(output):
        digest = sha256_file(path, MAX_TOTAL_OUTPUT_BYTES)
        lines.append(f"{digest}  {relative_output(path, output)}")
    atomic_write(output / "checksums.sha256", ("\n".join(lines) + "\n").encode("utf-8"))


def execute(
    *,
    runtimes: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    output: Path,
    manifest_sha256: str,
    spec_path: Path,
    spec_sha256: str | None,
    spec_identity: tuple[int, int, int, int] | None,
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    failed = False
    output_tree_error: str | None = None
    run_stop_reason: str | None = None
    runtime_ids = [runtime["id"] for runtime in runtimes]
    case_ids = [case["id"] for case in cases]
    stop_requested = False
    for case in cases:
        if stop_requested:
            break
        # Warmups and measured repetitions are interleaved across runtimes for
        # this case.  This reduces a simple first-runtime thermal/page-cache
        # advantage while preserving deterministic order.
        for warmup in range(case["warmups"]):
            if stop_requested:
                break
            for runtime in runtimes:
                phase = "warmup"
                location = trial_path(output, runtime["id"], case["id"], phase, warmup)
                env = dict(runtime["_resolved_env"])
                env.update(case["env"])
                record, did_fail = execute_one(
                    runtime=runtime,
                    case=case,
                    env=env,
                    phase=phase,
                    index=warmup,
                    location=location,
                    output=output,
                    manifest_sha256=manifest_sha256,
                )
                records.append(record)
                failed |= did_fail
                if record.get("capture_thread_leaked"):
                    failed = True
                    run_stop_reason = "capture reader thread did not terminate; stopped matrix"
                    stop_requested = True
                    break
        for repetition in range(case["repetitions"]):
            if stop_requested:
                break
            for runtime in runtimes:
                phase = "run"
                location = trial_path(output, runtime["id"], case["id"], phase, repetition)
                env = dict(runtime["_resolved_env"])
                env.update(case["env"])
                record, did_fail = execute_one(
                    runtime=runtime,
                    case=case,
                    env=env,
                    phase=phase,
                    index=repetition,
                    location=location,
                    output=output,
                    manifest_sha256=manifest_sha256,
                )
                records.append(record)
                failed |= did_fail
                process_wall = record.get("process_elapsed_s")
                wall_text = (
                    f"{process_wall:.6f}s"
                    if isinstance(process_wall, (int, float))
                    else "n/a"
                )
                print(
                    f"[{runtime['id']}/{case['id']}] run {repetition + 1}/{case['repetitions']} "
                    f"status={record['status']} process_wall={wall_text}",
                    flush=True,
                )
                if record.get("capture_thread_leaked"):
                    failed = True
                    run_stop_reason = "capture reader thread did not terminate; stopped matrix"
                    stop_requested = True
                    break
        try:
            check_output_tree_size(output, MAX_TOTAL_OUTPUT_BYTES)
        except (OSError, RuntimeError) as error:
            output_tree_error = str(error)
            failed = True
            break
    # Detect mutation of a file-backed specification after manifest creation.
    # The parsed spec is immutable for this run, so finalize the artifacts but
    # mark the archive failed rather than abandoning results/checksums.
    spec_changed = False
    if spec_identity is not None:
        try:
            spec_changed = (
                file_identity(spec_path) != spec_identity
                or (spec_sha256 is not None and sha256_file(spec_path) != spec_sha256)
            )
        except (OSError, RuntimeError):
            spec_changed = True
        if spec_changed:
            failed = True
    manifest_changed = False
    try:
        manifest_changed = sha256_file(output / "manifest.json") != manifest_sha256
        with open_regular_file(output / "manifest.sha256") as marker:
            manifest_changed = manifest_changed or (
                marker.read(256) != f"{manifest_sha256}  manifest.json\n".encode("ascii")
            )
    except (OSError, RuntimeError, UnicodeError):
        manifest_changed = True
    if manifest_changed:
        failed = True
    summaries, comparisons = summarize_records(records, runtime_ids, case_ids)
    results = {
        "schema": SCHEMA,
        "results_version": 1,
        "manifest_sha256": manifest_sha256,
        "completed_at": utc_now(),
        "spec_changed_during_run": spec_changed,
        "manifest_changed_during_run": manifest_changed,
        "output_tree_error": output_tree_error,
        "run_stop_reason": run_stop_reason,
        "records": records,
    }
    summary = {
        "schema": SCHEMA,
        "summary_version": 1,
        "manifest_sha256": manifest_sha256,
        "status": "complete" if not failed else "complete-with-failures",
        "spec_changed_during_run": spec_changed,
        "manifest_changed_during_run": manifest_changed,
        "output_tree_error": output_tree_error,
        "run_stop_reason": run_stop_reason,
        "completed_at": results["completed_at"],
        "comparison_policy": (
            "No runtime is designated as ground truth; output hashes and "
            "wall/resource metrics are descriptive."
        ),
        "runtime_case_summaries": summaries,
        "pairwise_comparisons": comparisons,
    }
    write_new_json(output / "results.json", results)
    write_new_json(output / "summary.json", summary)
    check_output_tree_size(output, MAX_TOTAL_OUTPUT_BYTES)
    write_checksums(output)
    return records, 1 if failed else 0


def execute_one(
    *,
    runtime: dict[str, Any],
    case: dict[str, Any],
    env: dict[str, str],
    phase: str,
    index: int,
    location: Path,
    output: Path,
    manifest_sha256: str,
) -> tuple[dict[str, Any], bool]:
    stdout_path = location / "stdout.bin"
    stderr_path = location / "stderr.bin"
    cwd = Path(runtime["cwd"])
    expected_identity = case.get("_runtime_identities", {}).get(runtime["id"])
    if expected_identity is None:
        expected_identity = {
            "cwd_identity": runtime["_cwd_identity"],
            "executable": runtime["_executable"],
            "executable_sha256": runtime["_executable_sha256"],
            "executable_identity": runtime.get("_executable_identity"),
        }
    trial = run_trial(
        argv=runtime["command"] + case["args"],
        cwd=cwd,
        env=env,
        timeout_s=case["timeout_s"],
        max_output_bytes=case["max_output_bytes"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        expected_identity=expected_identity,
    )
    trial.update(
        {
            "schema": SCHEMA,
            "runtime_id": runtime["id"],
            "case_id": case["id"],
            "phase": phase,
            "iteration": index,
            "warmup": phase == "warmup",
            "command": runtime["command"] + case["args"],
            "cwd": runtime["cwd"],
            "runtime_env_overrides": runtime["env"],
            "case_env_overrides": case["env"],
            "manifest_sha256": manifest_sha256,
            "stdout": {
                **trial["stdout"],
                "path": relative_output(stdout_path, output),
            },
            "stderr": {
                **trial["stderr"],
                "path": relative_output(stderr_path, output),
            },
        }
    )
    # A trial record is written only once and carries enough metadata to audit
    # it without trusting the aggregate results file.
    write_new_json(location / "trial.json", trial)
    return trial, trial["status"] != "ok"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON spec, or - for stdin")
    parser.add_argument("--output", type=Path, required=True, help="new output directory (must not exist)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec
    output = args.output
    try:
        if output.exists():
            raise SpecError(f"--output must not already exist: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        spec, raw_spec, raw_spec_sha256, spec_identity = read_spec(spec_path)
        spec_base = Path.cwd() if str(spec_path) == "-" else spec_path.resolve().parent
        normalized, runtimes, cases = validate_spec(spec, spec_base=spec_base)
        # Serialize the complete immutable manifest before creating the output
        # directory. A malformed value (including an unpaired JSON surrogate)
        # must fail preflight without leaving a misleading partial run.
        manifest = build_manifest(
            spec=spec,
            raw_spec_sha256=raw_spec_sha256,
            spec_path=spec_path,
            spec_identity=spec_identity,
            normalized=normalized,
            runtimes=runtimes,
            cases=cases,
            output=output,
        )
        manifest_bytes = pretty_json(manifest)
        output.mkdir()
        atomic_write(output / "manifest.json", manifest_bytes)
        manifest_sha256 = sha256_bytes(manifest_bytes)
        atomic_write(output / "manifest.sha256", f"{manifest_sha256}  manifest.json\n".encode("ascii"))
        print(f"manifest: {output / 'manifest.json'}")
        _, status = execute(
            runtimes=runtimes,
            cases=cases,
            output=output,
            manifest_sha256=manifest_sha256,
            spec_path=spec_path,
            spec_sha256=raw_spec_sha256,
            spec_identity=spec_identity,
        )
        print(f"summary: {output / 'summary.json'}")
        print(f"checksums: {output / 'checksums.sha256'}")
        return status
    except (
        SpecError,
        OSError,
        RuntimeError,
        UnicodeError,
        RecursionError,
        json.JSONDecodeError,
    ) as error:
        # A manifest may already exist.  Do not rewrite or delete it: a failed
        # run remains inspectable, while configuration errors fail before it.
        print(f"external benchmark failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
