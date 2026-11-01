"""
Sanitize identifying filesystem paths from text-based experiment artifacts.

Purpose
-------
This utility removes absolute local filesystem paths from historical result
files without changing numerical experiment values. It is intended for an
anonymous research artifact.

Default behavior is a dry run. Use --apply to overwrite files in place after
reviewing the report.

Examples
--------
Dry run:
    python src/sanitize_artifact_metadata.py

Apply changes:
    python src/sanitize_artifact_metadata.py --apply

Scan additional roots:
    python src/sanitize_artifact_metadata.py --apply results supplementary

Verification-only pass:
    python src/sanitize_artifact_metadata.py --verify-only
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {
    ".json",
    ".csv",
    ".txt",
    ".md",
    ".tex",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".tsv",
}

# Files and directories that should never be edited by this utility.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".venv",
    "venv",
    "env",
    "__pycache__",
}

# A project marker lets us convert historical absolute paths created on another
# machine into project-relative paths. The current project directory name is
# added dynamically at runtime.
DEFAULT_PROJECT_MARKERS = {
    "bitcoin-aml-transformer",
}

# Detect common identifying path forms.
UNIX_HOME_PATH_RE = re.compile(r"""(?P<path>/(?:home|Users)/[^/\s"'<>]+/[^\s"'<>]*)""")
WINDOWS_USER_PATH_RE = re.compile(
    r"""(?P<path>[A-Za-z]:\\Users\\[^\\\s"'<>]+\\[^\s"'<>]*)"""
)

# More general absolute paths are sanitized only when they clearly contain a
# known project marker. This avoids accidentally rewriting legitimate URLs or
# mathematical text.
UNIX_ABSOLUTE_WITH_MARKER_RE = re.compile(r"""(?P<path>/[^\s"'<>]*?)""")
WINDOWS_ABSOLUTE_WITH_MARKER_RE = re.compile(r"""(?P<path>[A-Za-z]:\\[^\s"'<>]*?)""")

EMAIL_RE = re.compile(
    r"""(?<![A-Za-z0-9._%+-])([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9._%+-])"""
)

# These patterns are reported, not automatically changed. Automatic deletion
# of arbitrary names or institutions could corrupt legitimate scientific text.
RISK_PATTERNS = {
    "Unix user path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.IGNORECASE),
    "Email address": EMAIL_RE,
}


@dataclass
class FileChange:
    path: Path
    replacements: int
    before_sha256: str
    after_sha256: str


def sha256_bytes(data: bytes) -> str:
    return __import__("hashlib").sha256(data).hexdigest()


def normalize_slashes(value: str) -> str:
    return value.replace("\\", "/")


def path_suffix_after_marker(path_text: str, markers: set[str]) -> str | None:
    """Return a project-relative suffix if a known project marker is present."""
    normalized = normalize_slashes(path_text)
    parts = [part for part in normalized.split("/") if part]

    for marker in markers:
        try:
            index = parts.index(marker)
        except ValueError:
            continue
        suffix_parts = parts[index + 1 :]
        if not suffix_parts:
            return "."
        return "/".join(suffix_parts)
    return None


def relative_to_current_project(path_text: str, project_root: Path) -> str | None:
    """Convert a path under the current project root to a relative POSIX path."""
    try:
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            return None
        return (
            candidate.resolve(strict=False)
            .relative_to(project_root.resolve(strict=False))
            .as_posix()
        )
    except (OSError, ValueError):
        return None


def sanitize_path_text(
    raw_path: str,
    *,
    project_root: Path,
    markers: set[str],
) -> str:
    """
    Convert an identifying absolute path to a project-relative path.

    If the historical path points inside this project (possibly on another
    machine), keep only the suffix after the project directory. If that cannot
    be inferred, retain only a neutral external basename.
    """
    current_relative = relative_to_current_project(raw_path, project_root)
    if current_relative is not None:
        return current_relative

    marker_relative = path_suffix_after_marker(raw_path, markers)
    if marker_relative is not None:
        return marker_relative

    # If a home-directory path cannot be mapped to the project, do not expose
    # the username. Preserve only the basename as a neutral external reference.
    normalized = normalize_slashes(raw_path).rstrip("/")
    basename = normalized.rsplit("/", 1)[-1] if normalized else "resource"
    return f"external/{basename}"


def replace_identifying_paths(
    text: str,
    *,
    project_root: Path,
    markers: set[str],
) -> tuple[str, int]:
    replacements = 0

    # A targeted token matcher is safer for JSON/CSV/TXT artifacts than a very
    # broad absolute-path regular expression. It catches the common historical
    # paths created under Linux/macOS and Windows user directories.
    token_re = re.compile(
        r"""(?P<path>
            /(?:home|Users)/[^,\s"'<>]+(?:/[^,\s"'<>]+)*
            |
            [A-Za-z]:\\Users\\[^,\s"'<>]+(?:\\[^,\s"'<>]+)*
        )""",
        re.VERBOSE,
    )

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        original = match.group("path")
        sanitized = sanitize_path_text(
            original,
            project_root=project_root,
            markers=markers,
        )
        if sanitized != original:
            replacements += 1
        return sanitized

    updated = token_re.sub(repl, text)
    return updated, replacements


def iter_text_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in TEXT_EXTENSIONS:
                yield root
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            yield path


def scan_risks(text: str) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []
    for label, pattern in RISK_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((label, match.group(0)))
    return findings


def sanitize_file(
    path: Path,
    *,
    project_root: Path,
    markers: set[str],
    apply: bool,
) -> FileChange | None:
    try:
        original_bytes = path.read_bytes()
        original_text = original_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    updated_text, replacements = replace_identifying_paths(
        original_text,
        project_root=project_root,
        markers=markers,
    )

    if replacements == 0 or updated_text == original_text:
        return None

    updated_bytes = updated_text.encode("utf-8")

    if apply:
        # Overwrite only the original file. No sensitive backup copy is created.
        path.write_bytes(updated_bytes)

    return FileChange(
        path=path,
        replacements=replacements,
        before_sha256=sha256_bytes(original_bytes),
        after_sha256=sha256_bytes(updated_bytes),
    )


def verify_roots(roots: Iterable[Path]) -> list[tuple[Path, str, str]]:
    findings: list[tuple[Path, str, str]] = []
    for path in iter_text_files(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, value in scan_risks(text):
            findings.append((path, label, value))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanitize identifying absolute filesystem paths from text-based "
            "experiment artifacts without changing numerical results."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=["results"],
        help=(
            "Files or directories to scan. Default: results. "
            "Only text-based artifact formats are processed."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Overwrite files in place. Without this option, perform a dry run.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not sanitize; only report residual user paths and email addresses.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root used to create relative paths. "
            "Default: parent of the src directory containing this script."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else script_path.parent.parent.resolve()
    )

    roots = [
        (project_root / root).resolve() if not Path(root).is_absolute() else Path(root)
        for root in args.roots
    ]

    markers = set(DEFAULT_PROJECT_MARKERS)
    markers.add(project_root.name)

    print("Artifact metadata sanitization")
    print("=============================")
    print(f"Project root : {project_root}")
    print(
        f"Mode         : {'VERIFY ONLY' if args.verify_only else ('APPLY' if args.apply else 'DRY RUN')}"
    )
    print("Scan roots   :")
    for root in roots:
        print(f"  - {root}")
    print()

    if args.verify_only:
        findings = verify_roots(roots)
        if not findings:
            print("Verification result: PASS")
            print("No user-directory paths or email addresses were found.")
            return 0

        print("Verification result: FAIL")
        print(f"Residual identifying patterns found: {len(findings)}")
        for path, label, value in findings[:200]:
            try:
                relative = path.relative_to(project_root)
            except ValueError:
                relative = path
            print(f"[{label}] {relative}: {value}")
        if len(findings) > 200:
            print(f"... {len(findings) - 200} additional findings omitted.")
        return 2

    changes: list[FileChange] = []
    scanned = 0

    for path in iter_text_files(roots):
        scanned += 1
        change = sanitize_file(
            path,
            project_root=project_root,
            markers=markers,
            apply=args.apply,
        )
        if change is not None:
            changes.append(change)

    print(f"Text files scanned : {scanned}")
    print(f"Files with changes : {len(changes)}")
    print(f"Path replacements  : {sum(item.replacements for item in changes)}")

    for item in changes:
        try:
            relative = item.path.relative_to(project_root)
        except ValueError:
            relative = item.path
        print(
            f"CHANGED {relative} "
            f"(replacements={item.replacements}, "
            f"before={item.before_sha256[:12]}, after={item.after_sha256[:12]})"
        )

    if not args.apply:
        print()
        print("Dry run only. No files were modified.")
        print("Review the list above, then rerun with --apply.")
        return 0

    print()
    print("Running post-sanitization verification...")
    findings = verify_roots(roots)

    if findings:
        print("Post-sanitization verification: FAIL")
        print(f"Residual identifying patterns found: {len(findings)}")
        for path, label, value in findings[:200]:
            try:
                relative = path.relative_to(project_root)
            except ValueError:
                relative = path
            print(f"[{label}] {relative}: {value}")
        if len(findings) > 200:
            print(f"... {len(findings) - 200} additional findings omitted.")
        return 2

    print("Post-sanitization verification: PASS")
    print(
        "No user-directory paths or email addresses remain in scanned text artifacts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
