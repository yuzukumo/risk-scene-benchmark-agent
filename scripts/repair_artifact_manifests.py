"""Refresh stale manifest hashes without modifying the referenced experiment results."""

import argparse
import json
from pathlib import Path

from nusc_scene_agent.artifact_manifest import build_artifact_entry, verify_artifact_manifest, write_artifact_manifest


def repair(path: Path) -> dict:
    validation = verify_artifact_manifest(path)
    if validation["valid"]:
        return validation
    payload = json.loads(path.read_text())
    entries = []
    for item in payload["artifacts"]:
        artifact = Path(item["path"])
        if not artifact.is_absolute():
            rooted = path.parent / artifact
            artifact = rooted if rooted.exists() or not artifact.exists() else artifact
        entries.append(build_artifact_entry(artifact, item["role"], item["kind"], path.parent))
    metadata = {**payload.get("metadata", {}), "previous_provenance": payload.get("provenance", {}),
                "repair_reason": validation["failures"], "repair_scope": "hash and file metadata only; results not recomputed"}
    write_artifact_manifest(path.parent, entries, metadata)
    return {"repaired": True, **verify_artifact_manifest(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    for manifest in args.manifests:
        print(json.dumps(repair(Path(manifest))))
