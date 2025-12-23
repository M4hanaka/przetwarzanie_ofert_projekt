# build_gold_pipeline.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pdf_to_text import PdfToTextConfig, pdf_bytes_to_text
from prompts import get_base_prompt, get_phase1_prompt, get_phase2_prompt
from llm_client import call_llama_structured
from models import OfferScan, OfferItem

# ---------------------------
# Konfiguracja
# ---------------------------

@dataclass(frozen=True)
class GoldPipelineConfig:
    input_pdf_dir: Path = Path("INPUT/oferty_pdf")
    output_offers_dir: Path = Path("DATASETS/offers")
    view_text_dir: Path = Path("OUTPUT/view_text")

    phase3_gold_dir: Path = Path("INPUT/phase3_gold")  # gdzie trzymasz Phase3 gold
    copy_phase3_gold: bool = True

    # Batch do Phase2 (datasetowy). 3 jest OK do debug, ale do datasetu zwykle za mało.
    phase2_batch_size: int = 20

    text_cfg: PdfToTextConfig = PdfToTextConfig(max_pages=8, max_chars=40_000, include_page_headers=True)

    # Kontrola generacji (opcjonalnie)
    phase1_max_new_tokens: int = 2500
    # phase2: nie ustawiamy sztywno; Outlines i safe-max-new-tokens w llm_client ogarnie
    phase2_max_new_tokens: Optional[int] = None


# ---------------------------
# I/O helpers
# ---------------------------

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _ensure_file_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


# ---------------------------
# Prompt builders (SFT-friendly)
# ---------------------------

def _wrap_pdf_text(pdf_text: str) -> str:
    return f"\n\n[PDF_TEXT_START]\n{pdf_text}\n[PDF_TEXT_END]\n"

def _build_phase1_user_content(pdf_text: str) -> str:
    base = get_base_prompt()
    p1 = get_phase1_prompt(base)
    return p1 + _wrap_pdf_text(pdf_text)

def _build_phase2_user_content(pdf_text: str, batch_items: List[Dict[str, Any]], batch_start: int) -> str:
    base = get_base_prompt()
    p2 = get_phase2_prompt(base, batch_items, batch_start)
    return p2 + _wrap_pdf_text(pdf_text)


# ---------------------------
# Batching
# ---------------------------

def _chunk(items: List[Any], batch_size: int) -> List[Tuple[int, List[Any]]]:
    out: List[Tuple[int, List[Any]]] = []
    for start in range(0, len(items), batch_size):
        out.append((start, items[start:start+batch_size]))
    return out


# ---------------------------
# Main per-offer run
# ---------------------------

def run_one_offer(pdf_path: Path, cfg: GoldPipelineConfig) -> None:
    offer_key = pdf_path.stem
    offer_dir = cfg.output_offers_dir / offer_key
    offer_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF -> TEXT
    pdf_bytes = pdf_path.read_bytes()
    pdf_text = pdf_bytes_to_text(pdf_bytes, cfg=cfg.text_cfg, ocr=None)

    # zapis podglądu (w dataset folderze + globalnie)
    (offer_dir / "pdf_text.txt").write_text(pdf_text, encoding="utf-8")
    cfg.view_text_dir.mkdir(parents=True, exist_ok=True)
    (cfg.view_text_dir / f"{offer_key}.txt").write_text(pdf_text, encoding="utf-8")

    # 2) Phase1 PRED
    phase1_pred_path = offer_dir / "phase1.pred.json"
    phase1_gold_path = offer_dir / "phase1.gold.json"

    phase1_user = _build_phase1_user_content(pdf_text)
    phase1_pred = call_llama_structured(
        phase1_user,
        OfferScan,
        max_new_tokens=cfg.phase1_max_new_tokens,
    )
    _write_json(phase1_pred_path, phase1_pred)

    # starter GOLD, jeżeli brak
    if not phase1_gold_path.exists():
        _write_json(phase1_gold_path, phase1_pred)

    # 3) Phase2 PRED per batch – UWAGA: batchujemy z GOLD Phase1 (a nie z PRED)
    phase1_gold = _read_json(phase1_gold_path)
    items_preview = phase1_gold.get("items_preview", []) or []
    if not isinstance(items_preview, list):
        raise ValueError("phase1.gold.json: items_preview must be a list")

    batches = _chunk(items_preview, cfg.phase2_batch_size)

    for batch_idx, (batch_start, batch_items) in enumerate(batches):
        batch_tag = f"{batch_idx:03d}"
        p2_pred_path = offer_dir / f"phase2.batch_{batch_tag}.pred.json"
        p2_gold_path = offer_dir / f"phase2.batch_{batch_tag}.gold.json"

        p2_user = _build_phase2_user_content(pdf_text, batch_items, batch_start)
        p2_pred = call_llama_structured(
            p2_user,
            list[OfferItem],
            max_new_tokens=cfg.phase2_max_new_tokens,
        )
        _write_json(p2_pred_path, p2_pred)

        if not p2_gold_path.exists():
            _write_json(p2_gold_path, p2_pred)

    # 4) Phase3 gold copy (opcjonalnie)
    if cfg.copy_phase3_gold:
        src = cfg.phase3_gold_dir / f"{offer_key}.json"
        dst = offer_dir / "phase3.gold.json"
        _ensure_file_copy(src, dst)


def run_all_offers(cfg: GoldPipelineConfig, *, only_missing: bool = False) -> None:
    pdf_files = sorted(cfg.input_pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Brak PDF w: {cfg.input_pdf_dir}")
        return

    for pdf_path in pdf_files:
        offer_key = pdf_path.stem
        offer_dir = cfg.output_offers_dir / offer_key
        phase1_gold_path = offer_dir / "phase1.gold.json"

        if only_missing and phase1_gold_path.exists():
            # jeśli Phase1 gold istnieje, zakładamy że oferta już była startowo przetworzona
            print(f"SKIP (exists): {offer_key}")
            continue

        print(f"\n=== RUN: {offer_key} ===")
        run_one_offer(pdf_path, cfg)


if __name__ == "__main__":
    cfg = GoldPipelineConfig(
        input_pdf_dir=Path("INPUT/oferty_pdf"),
        output_offers_dir=Path("DATASETS/offers"),
        view_text_dir=Path("OUTPUT/view_text"),
        phase3_gold_dir=Path("INPUT/phase3_gold"),
        phase2_batch_size=20,  # do datasetu lepiej 15-30 niż 3
    )
    run_all_offers(cfg, only_missing=False)
