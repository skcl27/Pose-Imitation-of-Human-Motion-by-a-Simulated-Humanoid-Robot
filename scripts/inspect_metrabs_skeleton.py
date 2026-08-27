"""One-off inspection script: print MeTRAbs' actual joint names/edges.

``src/perception/landmarks.py``'s ``CANONICAL_TO_RAW_ALIASES`` was written
without access to a live model/GPU. Run this ONCE on the target GPU machine
after installing the ``pose`` requirements to confirm (or correct) that
mapping before trusting any retargeting output.

Usage:
    python scripts/inspect_metrabs_skeleton.py [model_url] [skeleton]

With no arguments, uses the defaults from configs/default.yaml
(pose.model_url / pose.skeleton).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.perception import metrabs_model  # noqa: E402
from src.perception.landmarks import POSE_LANDMARKS, build_raw_to_canonical_map  # noqa: E402


def main() -> int:
    model_url = sys.argv[1] if len(sys.argv) > 1 else metrabs_model.DEFAULT_MODEL_URL
    skeleton = sys.argv[2] if len(sys.argv) > 2 else metrabs_model.DEFAULT_SKELETON

    print(f"Loading {model_url!r} ...")
    info = metrabs_model.skeleton_info(model_url, skeleton)
    print(f"\nSkeleton '{skeleton}' has {len(info.names)} joints:")
    for i, name in enumerate(info.names):
        print(f"  {i:2d}  {name}")
    print(f"\n{len(info.edges)} kinematic-tree edges (index pairs):")
    for a, b in info.edges:
        print(f"  {info.names[a]} -- {info.names[b]}")

    mapping = build_raw_to_canonical_map(list(info.names))
    matched = set(mapping.values())
    missing = [n for n in POSE_LANDMARKS if n not in matched]
    unmatched_raw = [n for n in info.names if n not in mapping]

    print("\n--- Alias-table check against src/perception/landmarks.py ---")
    if not missing and not unmatched_raw:
        print("OK: every canonical landmark matched, no leftover raw names.")
    if missing:
        print(f"MISSING canonical landmarks (not matched to any raw name): {missing}")
        print("-> add the real raw name(s) to CANONICAL_TO_RAW_ALIASES for these.")
    if unmatched_raw:
        print(f"Unmatched raw joint names (fine to ignore if unneeded): {unmatched_raw}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
