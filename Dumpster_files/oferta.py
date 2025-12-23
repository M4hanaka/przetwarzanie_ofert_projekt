#oferta.py
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

from processor import _pdf_bytes_to_text, BATCH_SIZE
from prompts import get_base_prompt, get_phase1_prompt, get_phase2_prompt

# === CONFIG ===

# Directory with input PDFs (one offer per file)
PDF_DIR = Path("oferty_pdf")

# Directory with "gold" JSONs: final target structure with top-level list,
# each element having keys: id, nazwa_dostawcy, data_oferty, pozycje_oferty.
GOLD_JSON_DIR = Path("gold_json")

# Output directory for SFT datasets
OUT_DIR = Path("train_json")

TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"

# Train/val split by *offers* (not by records)
TRAIN_FRACTION = 0.9
RANDOM_SEED = 42


def load_gold_offer(json_path: Path) -> Dict[str, Any]:
    """
    Loads a single-offer 'gold' JSON file.
    Expected format:

    [
      {
        "id": "...",
        "nazwa_dostawcy": "...",
        "data_oferty": "YYYY-MM-DD" or null,
        "pozycje_oferty": [ ... OfferItem ... ]
      }
    ]
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{json_path} contains an empty list")
        if len(data) > 1:
            # If there is more than one element, we treat the first as the main offer.
            # You can refine this if needed.
            offer = data[0]
        else:
            offer = data[0]
    elif isinstance(data, dict):
        offer = data
    else:
        raise ValueError(f"Unexpected JSON top-level type in {json_path}: {type(data)}")
    return offer

def build_phase1_target(offer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the OfferScan-like target for Phase 1 from a 'gold' offer.
    We re-create items_preview from pozycje_oferty, putting numer_oem
    back into the beginning of opis (because that's what Phase 1 is supposed to output).
    """
    items_preview: List[Dict[str, Any]] = []
    pozycje = offer.get("pozycje_oferty") or []
    for idx, item in enumerate(pozycje, start=1):
        opis = item.get("opis", "") or ""
        numer_oem = item.get("numer_oem", "") or ""

        # Rebuild Phase1-style opis that still includes OEM at the beginning
        if numer_oem:
            # Avoid doubling the OEM if it's already there
            normalized = opis.strip().lower()
            if not normalized.startswith(numer_oem.lower()):
                preview_opis = f"{numer_oem} {opis}".strip()
            else:
                preview_opis = opis
        else:
            preview_opis = opis

        items_preview.append(
            {
                "position_id": idx,
                "opis": preview_opis,
                "kwota_ceny_oferty": 0.0,  # we don't have unit prices in the gold JSON
            }
        )

    return {
        "id": offer.get("id", "") or "",
        "nazwa_dostawcy": offer.get("nazwa_dostawcy", "") or "",
        "data_oferty": offer.get("data_oferty", None),
        "items_preview": items_preview,
    }


def build_phase2_targets(offer: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the list of OfferItem dicts for Phase 2 from pozycje_oferty.
    """
    pozycje = offer.get("pozycje_oferty") or []
    out: List[Dict[str, Any]] = []

    for item in pozycje:
        out.append(
            {
                "id": str(item.get("id", "")),
                "id_oferty": item.get("id_oferty", "") or "",
                "opis": item.get("opis", "") or "",
                "grupa_materialow": item.get("grupa_materialow", "") or "",
                "numer_oem": item.get("numer_oem", "") or "",
                "producent": item.get("producent", "") or "",
                "serial_numbers": item.get("serial_numbers", "") or "",
            }
        )
    return out


def build_phase1_record(
    offer_id: str,
    pdf_text: str,
    offer: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a single SFT record for Phase 1 for this offer.
    """
    base_prompt = get_base_prompt()
    phase1_prompt = get_phase1_prompt(base_prompt)

    # >>> TU POWSTAJE user.content DLA PHASE1 <<<
    user_content = f"{phase1_prompt}\n\n{pdf_text}"
    target = build_phase1_target(offer)

    record = {
        "phase": "phase1",
        "offer_id": offer_id,
        "batch_index": None,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ],
    }
    return record


def build_phase2_records(
    offer_id: str,
    pdf_text: str,
    offer: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Create Phase 2 SFT records (one per batch) for this offer.
    """
    base_prompt = get_base_prompt()
    phase2_records: List[Dict[str, Any]] = []

    # First reconstruct phase1-style previews (we'll re-use them in the Phase 2 prompt)
    phase1_target = build_phase1_target(offer)
    preview_items = phase1_target["items_preview"]

    all_items = build_phase2_targets(offer)

    if len(all_items) != len(preview_items):
        # This shouldn't happen if your gold data is consistent,
        # but we guard against it to avoid silently misaligning items.
        raise ValueError(
            f"Mismatch between pozycje_oferty ({len(all_items)}) and items_preview ({len(preview_items)}) "
            f"for offer_id={offer_id}"
        )

    batch_index = 0
    start = 0
    n = len(all_items)

    while start < n:
        end = min(start + BATCH_SIZE, n)
        batch_preview = preview_items[start:end]
        batch_items = all_items[start:end]

        # Build Phase 2 prompt for this batch
        phase2_prompt = get_phase2_prompt(base_prompt, batch_preview, batch_start=start)

        # >>> TU POWSTAJE user.content DLA PHASE2 <<<
        user_content = f"{phase2_prompt}\n\n{pdf_text}"

        record = {
            "phase": "phase2",
            "offer_id": offer_id,
            "batch_index": batch_index,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": batch_items},
            ],
        }
        phase2_records.append(record)

        batch_index += 1
        start = end

    return phase2_records


def find_matching_json(pdf_path: Path) -> Path:
    """
    Given e.g. 'oferty_pdf/1_144_weidmuller.pdf',
    return 'LLM_output/1_144_weidmuller.json'.
    """
    stem = pdf_path.stem
    candidate = GOLD_JSON_DIR / f"{stem}.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No gold JSON found for {pdf_path} (expected {candidate})")


def gather_offers() -> List[Tuple[str, Path, Path]]:
    """
    Return a list of (offer_id, pdf_path, json_path) triples.

    For offer_id we first try OfferItem['id_oferty'] from gold JSON,
    otherwise we fall back to the file stem.
    """
    offers: List[Tuple[str, Path, Path]] = []

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        json_path = find_matching_json(pdf_path)
        offer = load_gold_offer(json_path)

        pozycje = offer.get("pozycje_oferty") or []
        offer_id = None
        if pozycje:
            # use id_oferty from the first item if available
            offer_id = pozycje[0].get("id_oferty") or None

        if not offer_id:
            # fall back to the top-level id or the file stem
            offer_id = offer.get("id") or pdf_path.stem

        offers.append((offer_id, pdf_path, json_path))

    return offers


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    offers = gather_offers()
    if not offers:
        print("No offers found. Check PDF_DIR and GOLD_JSON_DIR.")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(offers)

    split_idx = int(len(offers) * TRAIN_FRACTION)
    train_offers = offers[:split_idx]
    val_offers = offers[split_idx:]

    print(f"Total offers: {len(offers)}")
    print(f"Train offers: {len(train_offers)}  -> {TRAIN_PATH}")
    print(f"Val offers:   {len(val_offers)}  -> {VAL_PATH}")

    def write_records(offer_list: List[Tuple[str, Path, Path]], out_path: Path):
        with out_path.open("w", encoding="utf-8") as f_out:
            for offer_id, pdf_path, json_path in offer_list:
                # 1) Load PDF and extract text
                with pdf_path.open("rb") as f_pdf:
                    pdf_bytes = f_pdf.read()
                pdf_text = _pdf_bytes_to_text(pdf_bytes)

                # 2) Load 'gold' offer JSON
                offer = load_gold_offer(json_path)

                # 3) Build Phase 1 record
                rec1 = build_phase1_record(offer_id, pdf_text, offer)
                f_out.write(json.dumps(rec1, ensure_ascii=False) + "\n")

                # 4) Build Phase 2 records
                for rec2 in build_phase2_records(offer_id, pdf_text, offer):
                    f_out.write(json.dumps(rec2, ensure_ascii=False) + "\n")

    write_records(train_offers, TRAIN_PATH)
    write_records(val_offers, VAL_PATH)

    print("Done.")


if __name__ == "__main__":
    main()
