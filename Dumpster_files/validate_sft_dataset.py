import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

PHASES = {"phase1", "phase2"}

PHASE1_REQUIRED_KEYS = {"id", "nazwa_dostawcy", "data_oferty", "items_preview"}

PHASE2_REQUIRED_ITEM_KEYS = {
    "id",
    "id_oferty",
    "opis",
    "grupa_materialow",
    "numer_oem",
    "producent",
    "serial_numbers",
}


def _fail(errors: List[str], msg: str) -> None:
    errors.append(msg)


def _load_jsonl(path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"{path}:{lineno} invalid JSONL line: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno} record must be an object")
            records.append((lineno, obj))
    return records


def _find_last_assistant(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m
    return None


def _parse_assistant_json(
    path: Path, lineno: int, msg: Dict[str, Any], errors: List[str]
) -> Optional[Any]:
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        _fail(errors, f"{path}:{lineno} assistant.content missing/empty or not a string")
        return None
    try:
        return json.loads(content)
    except Exception as e:
        _fail(errors, f"{path}:{lineno} assistant.content is not valid JSON: {e}")
        return None


def validate_file(
    path: Path,
    *,
    strict_phase2_exact_keys: bool,
    require_assistant_last: bool,
    require_system: bool,
    enforce_batch_size: Optional[int],
) -> List[str]:
    errors: List[str] = []
    records = _load_jsonl(path)

    if not records:
        _fail(errors, f"{path} is empty (no records)")
        return errors

    for lineno, rec in records:
        # Basic top-level checks
        phase = rec.get("phase")
        if phase not in PHASES:
            _fail(errors, f"{path}:{lineno} invalid/missing phase: {phase!r}")

        offer_id = rec.get("offer_id")
        if offer_id is None or not isinstance(offer_id, str) or not offer_id.strip():
            _fail(errors, f"{path}:{lineno} missing/empty offer_id (must be non-empty string)")

        batch_index = rec.get("batch_index")
        if batch_index is None or not isinstance(batch_index, int) or batch_index < 0:
            _fail(errors, f"{path}:{lineno} missing/invalid batch_index (must be int >= 0)")

        messages = rec.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            _fail(errors, f"{path}:{lineno} messages missing/invalid (must be list with >= 2 items)")
            continue

        # Optional: require system message at position 0
        if require_system:
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "system":
                _fail(errors, f"{path}:{lineno} require_system enabled, but messages[0].role != 'system'")

        # Validate message objects
        for mi, m in enumerate(messages):
            if not isinstance(m, dict):
                _fail(errors, f"{path}:{lineno} messages[{mi}] is not an object")
                continue
            role = m.get("role")
            if role not in {"system", "user", "assistant"}:
                _fail(errors, f"{path}:{lineno} messages[{mi}].role invalid: {role!r}")
            if "content" not in m:
                _fail(errors, f"{path}:{lineno} messages[{mi}] missing content")

        # Find assistant message
        last_assistant = _find_last_assistant(messages)
        if last_assistant is None:
            _fail(errors, f"{path}:{lineno} missing assistant message")
            continue

        if require_assistant_last:
            if not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                _fail(errors, f"{path}:{lineno} require_assistant_last enabled, but last message is not assistant")

        # Parse assistant JSON (ground-truth)
        parsed = _parse_assistant_json(path, lineno, last_assistant, errors)
        if parsed is None:
            continue  # can't validate deeper

        # Phase-specific checks
        if phase == "phase1":
            if not isinstance(parsed, dict):
                _fail(errors, f"{path}:{lineno} phase1 assistant JSON must be an object")
                continue

            missing = PHASE1_REQUIRED_KEYS - set(parsed.keys())
            if missing:
                _fail(errors, f"{path}:{lineno} phase1 missing keys: {sorted(missing)}")

            # items_preview type check (if present)
            if "items_preview" in parsed and not isinstance(parsed["items_preview"], list):
                _fail(errors, f"{path}:{lineno} phase1 items_preview must be a list")

            # data_oferty can be None or str; accept both
            if "data_oferty" in parsed and parsed["data_oferty"] is not None and not isinstance(parsed["data_oferty"], str):
                _fail(errors, f"{path}:{lineno} phase1 data_oferty must be null or string")

        elif phase == "phase2":
            if not isinstance(parsed, list):
                _fail(errors, f"{path}:{lineno} phase2 assistant JSON must be a list")
                continue

            if enforce_batch_size is not None:
                if len(parsed) != enforce_batch_size:
                    _fail(
                        errors,
                        f"{path}:{lineno} phase2 batch size {len(parsed)} != expected {enforce_batch_size}",
                    )

            for ii, item in enumerate(parsed):
                if not isinstance(item, dict):
                    _fail(errors, f"{path}:{lineno} phase2 item[{ii}] must be an object")
                    continue

                # keys check
                item_keys = set(item.keys())
                if strict_phase2_exact_keys:
                    if item_keys != PHASE2_REQUIRED_ITEM_KEYS:
                        _fail(
                            errors,
                            f"{path}:{lineno} phase2 item[{ii}] keys mismatch. "
                            f"Expected exactly {sorted(PHASE2_REQUIRED_ITEM_KEYS)}, got {sorted(item_keys)}",
                        )
                else:
                    missing = PHASE2_REQUIRED_ITEM_KEYS - item_keys
                    if missing:
                        _fail(errors, f"{path}:{lineno} phase2 item[{ii}] missing keys: {sorted(missing)}")

                # id_oferty must be non-empty string
                id_oferty = item.get("id_oferty", "")
                if not isinstance(id_oferty, str) or not id_oferty.strip():
                    _fail(errors, f"{path}:{lineno} phase2 item[{ii}].id_oferty is empty/invalid")

                # Basic types for core fields (lightweight, practical)
                for k in PHASE2_REQUIRED_ITEM_KEYS:
                    if k not in item:
                        continue
                    if not isinstance(item[k], str):
                        _fail(errors, f"{path}:{lineno} phase2 item[{ii}].{k} must be a string")

        # else: phase missing/invalid already reported

    return errors


def main():
    default_train = "/home/karol_zych/przetwarzanie_ofert_projekt/INPUT/train_json/trainn.jsonl"
    default_val = "/home/karol_zych/przetwarzanie_ofert_projekt/INPUT/train_json/vall.jsonl"

    ap = argparse.ArgumentParser(
        description="Validate SFT JSONL dataset for Qwen offers (phase1/phase2)."
    )

    # paths jako opcjonalne; jeśli brak -> weź defaulty
    ap.add_argument(
        "paths",
        nargs="*",
        help="Paths to JSONL files (e.g., train.jsonl val.jsonl). If empty, defaults are used.",
    )
    ap.add_argument(
        "--strict-phase2-exact-keys",
        action="store_true",
        help="Require phase2 items to have exactly 7 keys (no extras).",
    )
    ap.add_argument(
        "--require-assistant-last",
        action="store_true",
        help="Require the last message in messages[] to be assistant.",
    )
    ap.add_argument(
        "--require-system",
        action="store_true",
        help="Require messages[0] to be a system message.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="If set, require phase2 batches to have exactly this many items.",
    )

    args = ap.parse_args()

    paths = args.paths
    if not paths:
        paths = [default_train, default_val]
        print("[INFO] No paths provided. Using defaults:")
        for p in paths:
            print("  -", p)

    any_errors = False
    for p in paths:
        path = Path(p)

        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            any_errors = True
            continue

        if not path.is_file():
            print(f"[ERROR] Not a file: {path}")
            any_errors = True
            continue

        # Szybka diagnostyka uprawnień / odczytu
        try:
            with path.open("r", encoding="utf-8") as _:
                pass
        except Exception as e:
            print(f"[ERROR] Cannot open file: {path} ({type(e).__name__}: {e})")
            any_errors = True
            continue

        errors = validate_file(
            path,
            strict_phase2_exact_keys=args.strict_phase2_exact_keys,
            require_assistant_last=args.require_assistant_last,
            require_system=args.require_system,
            enforce_batch_size=args.batch_size,
        )

        if errors:
            any_errors = True
            print(f"\n[FAIL] {path} ({len(errors)} issues)")
            for e in errors[:200]:
                print(" -", e)
            if len(errors) > 200:
                print(f" ... truncated, total issues: {len(errors)}")
        else:
            print(f"[OK] {path} passed validation")

    raise SystemExit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
