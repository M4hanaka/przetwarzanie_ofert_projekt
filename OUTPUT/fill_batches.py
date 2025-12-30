# -*- coding: utf-8 -*-
"""
Fill phase2.batch_XXX.pred.json files by splitting items from <folder_name>.json (master offer file).

Assumptions:
- Root directory contains many offer folders.
- In each offer folder, there is a "master" JSON file named exactly like the folder + ".json".
- Master JSON structure is a list of offer objects, each with "pozycje_oferty" (list of items).
- We split items into chunks of size 3 and write them into phase2.batch_000.pred.json, etc.

Safety:
- By default: creates backups of overwritten phase2 batch files.
- Can optionally prune extra phase2.batch_*.pred.json that are no longer needed.

Usage:
  python fill_phase2_batches_from_master.py --root "/home/karol_zych/przetwarzanie_ofert_projekt/OUTPUT/preds copy"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BATCH_SIZE_DEFAULT = 3
BATCH_PRED_RE = re.compile(r"^phase2\.batch_(\d{3})\.pred\.json$")


@dataclass(frozen=True)
class Options:
    root: Path
    batch_size: int = BATCH_SIZE_DEFAULT
    dry_run: bool = False
    backup: bool = True
    prune_extras: bool = False
    encoding: str = "utf-8"


def read_json(path: Path, encoding: str = "utf-8") -> Any:
    with path.open("r", encoding=encoding) as f:
        return json.load(f)


def write_json(path: Path, data: Any, encoding: str = "utf-8") -> None:
    # Ensure stable, readable JSON
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def collect_items_from_master(master_data: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - a list of offer dicts (your example)
      - OR a single offer dict (in case some files are not wrapped in a list)
    Returns:
      - concatenated list of all "pozycje_oferty"
    """
    offers: List[Dict[str, Any]]
    if isinstance(master_data, dict):
        offers = [master_data]
    elif isinstance(master_data, list):
        offers = [x for x in master_data if isinstance(x, dict)]
    else:
        return []

    all_items: List[Dict[str, Any]] = []
    for offer in offers:
        items = offer.get("pozycje_oferty", [])
        if isinstance(items, list):
            all_items.extend([it for it in items if isinstance(it, dict)])

    return all_items


def list_existing_batch_pred_files(folder: Path) -> List[Tuple[int, Path]]:
    found: List[Tuple[int, Path]] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = BATCH_PRED_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return found


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    # Overwrite old backup to keep it simple and deterministic
    shutil.copy2(path, bak)
    return bak


def process_offer_folder(folder: Path, opt: Options) -> Dict[str, Any]:
    """
    Returns a result dict for logging/reporting.
    """
    folder_name = folder.name
    master_path = folder / f"{folder_name}.json"

    result: Dict[str, Any] = {
        "folder": str(folder),
        "master_path": str(master_path),
        "status": "ok",
        "items_total": 0,
        "batches_needed": 0,
        "written": [],
        "skipped_reason": None,
        "pruned": [],
    }

    if not master_path.exists():
        result["status"] = "skipped"
        result["skipped_reason"] = "missing_master_json"
        return result

    try:
        master_data = read_json(master_path, encoding=opt.encoding)
    except Exception as e:
        result["status"] = "error"
        result["skipped_reason"] = f"master_read_error: {e}"
        return result

    items = collect_items_from_master(master_data)
    result["items_total"] = len(items)

    if len(items) == 0:
        result["status"] = "skipped"
        result["skipped_reason"] = "no_items_in_master"
        return result

    chunks = chunk_list(items, opt.batch_size)
    result["batches_needed"] = len(chunks)

    # Write needed batches
    for idx, chunk in enumerate(chunks):
        out_path = folder / f"phase2.batch_{idx:03d}.pred.json"

        if opt.dry_run:
            result["written"].append({"path": str(out_path), "count": len(chunk), "dry_run": True})
            continue

        if opt.backup and out_path.exists():
            backup_file(out_path)

        try:
            write_json(out_path, chunk, encoding=opt.encoding)
            result["written"].append({"path": str(out_path), "count": len(chunk), "dry_run": False})
        except Exception as e:
            result["status"] = "error"
            result["skipped_reason"] = f"write_error: {e}"
            return result

    # Optionally prune extra batch pred files that are no longer needed
    if opt.prune_extras and not opt.dry_run:
        existing = list_existing_batch_pred_files(folder)
        needed_indices = set(range(len(chunks)))
        for i, p in existing:
            if i not in needed_indices:
                try:
                    if opt.backup:
                        backup_file(p)
                    p.unlink()
                    result["pruned"].append(str(p))
                except Exception as e:
                    result["status"] = "error"
                    result["skipped_reason"] = f"prune_error: {e}"
                    return result

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to 'preds copy' directory")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT, help="Items per batch (default: 3)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files, only print plan")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when overwriting")
    ap.add_argument("--prune-extras", action="store_true", help="Delete extra phase2.batch_*.pred.json beyond what is needed")
    args = ap.parse_args()

    opt = Options(
        root=Path(args.root).expanduser().resolve(),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        backup=(not args.no_backup),
        prune_extras=args.prune_extras,
    )

    if not opt.root.exists() or not opt.root.is_dir():
        raise SystemExit(f"[ERROR] root does not exist or is not a directory: {opt.root}")

    folders = [p for p in opt.root.iterdir() if p.is_dir()]
    folders.sort(key=lambda p: p.name.lower())

    summary = {
        "root": str(opt.root),
        "folders_total": len(folders),
        "processed_ok": 0,
        "skipped": 0,
        "errors": 0,
        "details": [],
    }

    for folder in folders:
        res = process_offer_folder(folder, opt)
        summary["details"].append(res)

        if res["status"] == "ok":
            summary["processed_ok"] += 1
        elif res["status"] == "skipped":
            summary["skipped"] += 1
        else:
            summary["errors"] += 1

    # Print concise report
    print("============================================================")
    print("PHASE2 BATCH FILL REPORT")
    print("============================================================")
    print(f"Root: {summary['root']}")
    print(f"Folders: {summary['folders_total']} | OK: {summary['processed_ok']} | Skipped: {summary['skipped']} | Errors: {summary['errors']}")
    print("------------------------------------------------------------")

    for d in summary["details"]:
        name = Path(d["folder"]).name
        if d["status"] == "ok":
            print(f"[OK] {name}: items={d['items_total']} -> batches={d['batches_needed']}")
        elif d["status"] == "skipped":
            print(f"[SKIP] {name}: {d['skipped_reason']}")
        else:
            print(f"[ERR] {name}: {d['skipped_reason']}")

    # Optional: save JSON report in root
    if not opt.dry_run:
        report_path = opt.root / "phase2_fill_report.json"
        write_json(report_path, summary)
        print("------------------------------------------------------------")
        print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
