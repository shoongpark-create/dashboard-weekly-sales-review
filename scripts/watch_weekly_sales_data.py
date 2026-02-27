#!/usr/bin/env python3
"""Watch Weekly_Sales_Review workbook and refresh date-partitioned CSV datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from prepare_weekly_sales_data import build_latest_outputs, source_fingerprint


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def make_snapshot_dir(history_root: Path) -> Path:
    now = datetime.now()
    date_dir = history_root / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    base = now.strftime("%H%M%S")
    candidate = date_dir / base
    seq = 1
    while candidate.exists():
        candidate = date_dir / f"{base}_{seq:02d}"
        seq += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def copy_latest_snapshot(
    latest_root: Path, snapshot_dir: Path, input_file: Path, fp: dict
) -> None:
    if latest_root.exists():
        shutil.copytree(latest_root, snapshot_dir / "latest", dirs_exist_ok=False)

    shutil.copy2(input_file, snapshot_dir / input_file.name)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(input_file),
        "source_fingerprint": fp,
    }
    (snapshot_dir / "snapshot_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def latest_uploaded_xlsx(uploads_dir: Path) -> Path | None:
    if not uploads_dir.exists():
        return None
    files = [
        p
        for p in uploads_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".xlsx" and not p.name.startswith("~$")
    ]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime_ns)


def maybe_sync_uploaded_file(
    input_path: Path, uploads_dir: Path, state: dict
) -> tuple[dict, bool]:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    latest = latest_uploaded_xlsx(uploads_dir)
    if latest is None:
        return state, False

    upload_record = {
        "path": str(latest),
        "fingerprint": source_fingerprint(latest),
    }
    if state.get("last_upload") == upload_record:
        return state, False

    input_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, input_path)
    state["last_upload"] = upload_record
    print(f"Synced uploaded file: {latest} -> {input_path}")
    return state, True


def process_once(
    input_path: Path,
    processed_root: Path,
    history_root: Path,
    state_file: Path,
    uploads_dir: Path,
) -> bool:
    state = load_state(state_file)
    state, upload_synced = maybe_sync_uploaded_file(input_path, uploads_dir, state)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    current_fp = source_fingerprint(input_path)
    previous_fp = state.get("source_fingerprint")
    changed = previous_fp != current_fp

    if not changed:
        if upload_synced:
            save_state(state_file, state)
        return False

    generated = build_latest_outputs(input_path, processed_root)
    snapshot_dir = make_snapshot_dir(history_root)
    copy_latest_snapshot(
        Path(generated["latest_root"]), snapshot_dir, input_path, current_fp
    )

    state.update(
        {
            "source_file": str(input_path),
            "source_fingerprint": current_fp,
            "last_processed_at": datetime.now().isoformat(timespec="seconds"),
            "last_snapshot_dir": str(snapshot_dir),
            "latest_root": generated["latest_root"],
        }
    )
    save_state(state_file, state)

    print(f"Updated latest files in: {generated['latest_root']}")
    print(f"Saved dated snapshot in: {snapshot_dir}")
    return True


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Watch Weekly_Sales_Review workbook and create dated CSV snapshots."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "Data" / "Weekly_Sales_Review_W6.xlsx",
        help="Path to source xlsx file.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=root / "Data" / "processed",
        help="Root directory for latest processed files.",
    )
    parser.add_argument(
        "--uploads-dir",
        type=Path,
        default=root / "Data" / "uploads",
        help="Directory to watch for newly uploaded xlsx files.",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=root / "Data" / "processed" / "history",
        help="Directory to store dated snapshots.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=root / "Data" / "processed" / ".watch_state.json",
        help="State file path for change tracking.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        changed = process_once(
            input_path=args.input,
            processed_root=args.processed_root,
            history_root=args.history_root,
            state_file=args.state_file,
            uploads_dir=args.uploads_dir,
        )
        if not changed:
            print("No changes detected.")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
