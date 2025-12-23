# build_sft_jsonl.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from prompts import get_base_prompt, get_phase1_prompt, get_phase2_prompt

SYSTEM_PROMPT = (
    "You extract structured information from PDF offer text. "
    "Focus on meaning. Return only JSON."
)

@dataclass(frozen=True)
class SFTBuildConfig:
    offers_dir: Path = Path("DATASETS/offers")
    out_train_jsonl: Path = Path("DATASETS/train.jsonl")
    out_val_jsonl: Path = Path("DATASETS/val.jsonl")

    # split na ofertach
    val_ratio: float = 0.15
    seed: int = 42

    include_system_message: bool = True


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _wrap_pdf_text(pdf_text: str) -> str:
    return f"\n\n[PDF_TEXT_START]\n{pdf_text}\n[PDF_TEXT_END]\n"

def _assistant_json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _build_phase1_user(pdf_text: str) -> str:
    base = get_base_prompt()
    p1 = get_phase1_prompt(base)
    return p1 + _wrap_pdf_text(pdf_text)

def _build_phase2_user(pdf_text: str, batch_items: List[Dict[str, Any]], batch_start: int) -> str:
    base = get_base_prompt()
    p2 = get_phase2_prompt(base, batch_items, batch_start)
    return p2 + _wrap_pdf_text(pdf_text)

def _load_phase2_batches(offer_dir: Path) -> List[Tuple[int, List[Dict[str, Any]]]]:
    # zwraca [(batch_index, gold_array), ...] posortowane po index
    batch_files = sorted(offer_dir.glob("phase2.batch_*.gold.json"))
    batches = []
    for bf in batch_files:
        # phase2.batch_000.gold.json -> 0
        idx_str = bf.name.split("batch_")[1].split(".")[0]
        batch_idx = int(idx_str)
        batches.append((batch_idx, _read_json(bf)))
    batches.sort(key=lambda x: x[0])
    return batches

def build_records_for_offer(cfg: SFTBuildConfig, offer_dir: Path) -> List[Dict[str, Any]]:
    offer_key = offer_dir.name
    pdf_text_path = offer_dir / "pdf_text.txt"
    phase1_gold_path = offer_dir / "phase1.gold.json"
    phase3_gold_path = offer_dir / "phase3.gold.json"

    if not pdf_text_path.exists():
        raise FileNotFoundError(f"{offer_key}: missing pdf_text.txt")
    if not phase1_gold_path.exists():
        raise FileNotFoundError(f"{offer_key}: missing phase1.gold.json")

    pdf_text = _read_text(pdf_text_path)
    phase1_gold = _read_json(phase1_gold_path)

    records: List[Dict[str, Any]] = []

    # ---- Phase1 record
    phase1_user = _build_phase1_user(pdf_text)
    phase1_assistant = _assistant_json_text(phase1_gold)

    messages = []
    if cfg.include_system_message:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": phase1_user})
    messages.append({"role": "assistant", "content": phase1_assistant})

    records.append({"messages": messages, "meta": {"offer_key": offer_key, "phase": "phase1"}})

    # ---- Phase2 records (batching prompt zależy od Phase1 GOLD items_preview)
    items_preview = phase1_gold.get("items_preview", []) or []
    # batch_start musi odpowiadać temu, co było użyte przy tworzeniu goldów
    # Używamy tego samego indeksowania jak w generatorze pred/gold: start=0, batch_size=...
    # Ale tu nie znamy batch_size z generatora – więc rekonstruujemy na podstawie liczby itemów w pliku.
    # W praktyce najbezpieczniej: trzymać batch_size w metadanych; na razie robimy "odczyt batchów" i liczymy start kumulacyjnie.

    phase2_batches = _load_phase2_batches(offer_dir)
    cum_start = 0
    for batch_idx, batch_gold_array in phase2_batches:
        if not isinstance(batch_gold_array, list):
            raise ValueError(f"{offer_key}: phase2 batch {batch_idx} gold is not a list")

        # batch_items do promptu bierzemy z Phase1 GOLD w tym samym zakresie
        batch_items = items_preview[cum_start:cum_start + len(batch_gold_array)]
        phase2_user = _build_phase2_user(pdf_text, batch_items, cum_start)

        phase2_assistant = _assistant_json_text(batch_gold_array)

        m2 = []
        if cfg.include_system_message:
            m2.append({"role": "system", "content": SYSTEM_PROMPT})
        m2.append({"role": "user", "content": phase2_user})
        m2.append({"role": "assistant", "content": phase2_assistant})

        records.append(
            {
                "messages": m2,
                "meta": {
                    "offer_key": offer_key,
                    "phase": "phase2",
                    "batch_index": batch_idx,
                    "batch_start": cum_start,
                    "items_in_batch": len(batch_gold_array),
                },
            }
        )

        cum_start += len(batch_gold_array)

    # ---- Phase3 record (opcjonalnie)
    if phase3_gold_path.exists():
        phase3_gold = _read_json(phase3_gold_path)

        # Tu wstawiasz swój prompt phase3 (skoro „OBECNIE TO MAM JUŻ”)
        # Na razie placeholder:
        phase3_user = "PHASE 3 TASK: Produce the final consolidated JSON.\n" + _wrap_pdf_text(pdf_text)

        m3 = []
        if cfg.include_system_message:
            m3.append({"role": "system", "content": SYSTEM_PROMPT})
        m3.append({"role": "user", "content": phase3_user})
        m3.append({"role": "assistant", "content": _assistant_json_text(phase3_gold)})

        records.append({"messages": m3, "meta": {"offer_key": offer_key, "phase": "phase3"}})

    return records


def main() -> None:
    cfg = SFTBuildConfig()

    offer_dirs = sorted([p for p in cfg.offers_dir.iterdir() if p.is_dir()])
    if not offer_dirs:
        raise FileNotFoundError(f"Brak ofert w {cfg.offers_dir}")

    rnd = random.Random(cfg.seed)
    offer_keys = [d.name for d in offer_dirs]
    rnd.shuffle(offer_keys)

    n_val = max(1, int(len(offer_keys) * cfg.val_ratio))
    val_set = set(offer_keys[:n_val])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for offer_dir in offer_dirs:
        records = build_records_for_offer(cfg, offer_dir)
        if offer_dir.name in val_set:
            val_records.extend(records)
        else:
            train_records.extend(records)

    cfg.out_train_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with cfg.out_train_jsonl.open("w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with cfg.out_val_jsonl.open("w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] train records: {len(train_records)} -> {cfg.out_train_jsonl}")
    print(f"[OK] val records:   {len(val_records)} -> {cfg.out_val_jsonl}")
    print(f"[OK] val offers: {len(val_set)}/{len(offer_dirs)}")


if __name__ == "__main__":
    main()
