"""Phase 8 — assemble a clean HF Docker Space folder and push it to the Hub.

The Space repo must have Dockerfile + README (with `sdk: docker` frontmatter) at
its root, alongside the code/artifacts the Dockerfile COPYs. This script stages
exactly those files into deploy/_space_build/ then uploads the folder.
`upload_folder` auto-uses Git LFS for the large *.keras file (per .gitattributes),
so no manual `git lfs` wrangling is needed.

Prereqs:
  1. pip install huggingface_hub   (already added to the venv)
  2. huggingface-cli login          (paste a WRITE token from hf.co/settings/tokens)

Usage:
  ./.venv/Scripts/python.exe deploy/deploy_hf_space.py --repo-id <username>/pneumoscan-api
Options:
  --private     create the Space private (default: public)
  --stage-only  assemble deploy/_space_build/ but do not create/upload the Space
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
STAGE_DIR = DEPLOY_DIR / "_space_build"

# Space-root files (from deploy/) + repo subsets the Dockerfile needs.
MODEL_FILES = [
    "models/best_model.keras",
    "models/threshold.json",
    "models/temperature.json",
    "models/triage.json",
]


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def build_stage() -> Path:
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True)

    # Space-root files.
    for name in ("Dockerfile", "README.md", ".gitattributes"):
        shutil.copy2(DEPLOY_DIR / name, STAGE_DIR / name)

    # Code the Dockerfile COPYs.
    _copy_tree(REPO_ROOT / "src", STAGE_DIR / "src")
    _copy_tree(REPO_ROOT / "api", STAGE_DIR / "api")

    # Only the artifacts the API loads at runtime (not the eval CSVs/jsons).
    (STAGE_DIR / "models").mkdir()
    missing = []
    for rel in MODEL_FILES:
        src = REPO_ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        shutil.copy2(src, STAGE_DIR / rel)
    if missing:
        sys.exit(f"ERROR: missing model artifacts: {missing}")

    print(f"Staged Space folder at: {STAGE_DIR}")
    for p in sorted(STAGE_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(STAGE_DIR).as_posix():40s} {p.stat().st_size:>10,} B")
    return STAGE_DIR


def push(repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi, whoami

    try:
        user = whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login  (needs a WRITE token)")
    print(f"Logged in as: {user}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                    private=private, exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(STAGE_DIR),
        commit_message="Deploy PneumoScan API (Phase 8)",
    )
    print("\nUpload complete. The Space will now build the Docker image (watch the")
    print(f"'Logs' tab). Once green, the API base URL is:")
    print(f"  https://{repo_id.replace('/', '-')}.hf.space")
    print("Verify:  curl https://<space-subdomain>.hf.space/health")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy PneumoScan API to a HF Docker Space.")
    ap.add_argument("--repo-id", help="e.g. yourname/pneumoscan-api")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()

    build_stage()
    if args.stage_only:
        print("\n--stage-only: skipped create/upload.")
        return
    if not args.repo_id:
        sys.exit("Provide --repo-id <username>/pneumoscan-api (or use --stage-only).")
    push(args.repo_id, args.private)


if __name__ == "__main__":
    main()
