from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ARTIFACT_MANIFEST_SCHEMA = "benchmark_artifact_manifest_v1"


@dataclass
class ArtifactEntry:
    path: str
    role: str
    kind: str
    format: str
    exists: bool
    size_bytes: int = 0
    sha256: str = ""
    modified_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_artifact_entry(path: Path, role: str, kind: str, output_root: Path | None = None) -> ArtifactEntry:
    path = Path(path)
    output_root = Path(output_root) if output_root is not None else None
    exists = path.exists()
    display_path = str(path)
    if output_root is not None:
        try:
            display_path = str(path.relative_to(output_root))
        except ValueError:
            display_path = str(path)
    return ArtifactEntry(
        path=display_path,
        role=role,
        kind=kind,
        format=_artifact_format(path),
        exists=exists,
        size_bytes=path.stat().st_size if exists and path.is_file() else 0,
        sha256=_sha256_file(path) if exists and path.is_file() else "",
        modified_at_utc=(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if exists
            else ""
        ),
    )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: List[str]) -> str:
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def collect_runtime_provenance() -> Dict[str, Any]:
    package_versions: Dict[str, str] = {}
    for package in ["numpy", "pandas", "torch", "nuscenes-devkit", "langgraph", "PyYAML"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue

    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    git_status = _command_output(["git", "status", "--porcelain"])
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    }


def write_artifact_manifest(
    output_dir: Path,
    artifacts: Iterable[ArtifactEntry],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [artifact.to_dict() for artifact in artifacts]
    payload = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "metadata": dict(metadata or {}),
        "provenance": collect_runtime_provenance(),
        "overview": {
            "artifact_count": len(entries),
            "existing_artifact_count": sum(1 for item in entries if bool(item.get("exists"))),
            "total_size_bytes": sum(int(item.get("size_bytes") or 0) for item in entries),
        },
        "artifacts": entries,
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "artifact_manifest.md").write_text(_render_manifest_markdown(payload), encoding="utf-8")
    return payload


def _artifact_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix:
        return suffix
    if path.is_dir():
        return "directory"
    return "unknown"


def verify_artifact_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    checked, failures = 0, []
    for entry in payload.get("artifacts", []):
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            rooted = manifest_path.parent / path
            path = rooted if rooted.exists() or not path.exists() else path
        if not entry.get("exists"):
            continue
        if not path.exists():
            failures.append({"path": str(path), "reason": "missing"})
        elif path.is_file() and entry.get("sha256"):
            checked += 1
            if path.stat().st_size != entry.get("size_bytes") or _sha256_file(path) != entry["sha256"]:
                failures.append({"path": str(path), "reason": "content_mismatch"})
    return {"manifest": str(manifest_path), "checked_files": checked, "valid": not failures, "failures": failures}


def _render_manifest_markdown(payload: Mapping[str, Any]) -> str:
    overview = dict(payload.get("overview") or {})
    lines: List[str] = [
        "# Artifact Manifest",
        "",
        f"- Artifacts: `{overview.get('artifact_count', 0)}`",
        f"- Existing artifacts: `{overview.get('existing_artifact_count', 0)}`",
        f"- Total size bytes: `{overview.get('total_size_bytes', 0)}`",
        "",
        "| Path | Role | Kind | Format | Exists | Size Bytes | SHA256 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in payload.get("artifacts", []):
        lines.append(
            "| `{0}` | `{1}` | `{2}` | `{3}` | `{4}` | `{5}` | `{6}` |".format(
                item.get("path", ""),
                item.get("role", ""),
                item.get("kind", ""),
                item.get("format", ""),
                item.get("exists", False),
                item.get("size_bytes", 0),
                item.get("sha256", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)
