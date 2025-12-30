# processor.py
from __future__ import annotations

from typing import Tuple, List, Dict, Any, Optional
import re

from models import OfferItem, OfferScan
from prompts import get_base_prompt, get_phase1_prompt, get_phase2_prompt
from llm_client import call_llama_structured, save_json

from pdf_to_text import pdf_bytes_to_text, PdfToTextConfig, OcrEngine
from pathlib import Path
import json, hashlib
from datetime import datetime
# ---------------------------
# Ustawienia
# ---------------------------

BATCH_SIZE = 3
ARTIFACTS_DIR = Path("OUTPUT/preds")
SAVE_PHASE_ARTIFACTS = True
PROMPT_VERSION = "prompts_v1"  # ustaw ręcznie i zmieniaj gdy edytujesz prompts.py


# Simple heuristics for OEM and classification codes (generic, not vendor-specific)
OEM_FROM_DESC_REGEX = re.compile(r"^([0-9A-Z][0-9A-Z0-9\-\/\. ]{1,80}?)\s*-\s+")
CN_REGEX = re.compile(r"\bCN[:\s]*([0-9]{4,})", re.IGNORECASE)
PKWIU_REGEX = re.compile(r"\bPKWIU[:\s]*([\d\.]+)", re.IGNORECASE)




def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def _offer_dir(offer_key: str) -> Path:
    d = ARTIFACTS_DIR / offer_key
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------------------------
# Status tracker
# ---------------------------

class StatusTracker:
    """Simple status tracker for progress updates."""
    def __init__(self):
        self.progress = 0
        self.message = ""
        self.step = 1

    def update_status(self, progress: int, message: str, step: int = 1):
        self.progress = progress
        self.message = message
        self.step = step
        print(f"[{progress}%] {message}")


def _update_progress(status_tracker: Optional[StatusTracker], progress: int, message: str, step: int = 1):
    if status_tracker:
        status_tracker.update_status(progress, message, step)


# ---------------------------
# OEM / kody klasyfikacyjne
# ---------------------------

def _looks_like_classification_code(text: str) -> bool:
    """
    True jeśli text wygląda jak kod klasyfikacyjny (CN, PKWiU itp.),
    a nie typowy numer OEM.
    """
    if not text:
        return False
    t = text.strip().upper()
    if t.startswith("CN") or t.startswith("PKWIU"):
        return True

    # bardzo prosta heurystyka: długi czysto-cyfrowy token
    digits_only = "".join(c for c in t if c.isdigit())
    if len(digits_only) >= 6 and len(digits_only) == len(t.replace(" ", "")):
        return True

    return False


def _guess_oem_from_description(opis: str) -> str:
    """
    Heurystycznie wyciąga numer OEM z początku opisu.
    Typowy pattern: 'CODE - opis...'
    """
    if not opis:
        return ""

    text = opis.strip().upper()

    # Case 1: CODE - ...
    m = OEM_FROM_DESC_REGEX.match(text)
    if m:
        return m.group(1).strip()

    # Case 2: pierwszy token wygląda jak kod (mieszanka liter i cyfr)
    first_token = text.split()[0]
    has_letter = any(c.isalpha() for c in first_token)
    has_digit = any(c.isdigit() for c in first_token)
    if has_letter and has_digit and len(first_token) >= 2:
        return first_token.strip()

    return ""

OFFER_ID_PATTERNS = [
    # OF/PiT/MMG/2025/03748 itp.
    re.compile(r"(?i)\b(OF/[A-Z0-9]+(?:/[A-Z0-9]+){2,})\b"),
    # 23/0717, 198/12/2024, 001399244 bywa bez separatorów
    re.compile(r"(?i)\b(\d{1,6}/\d{1,6}(?:/\d{1,6})?)\b"),
    # Oferta Nr 21004439 / Offer No.: 23/0717 / Numer oferty: ...
    re.compile(r"(?i)\b(?:oferta\s*nr|offer\s*no\.?|numer\s*oferty|nr)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-\._]{2,40})\b"),
]

def _extract_offer_id_from_pdf_text(pdf_text: str) -> str:
    if not pdf_text:
        return ""
    text = str(pdf_text)

    for rx in OFFER_ID_PATTERNS:
        m = rx.search(text)
        if m:
            return m.group(1).strip()
    return ""


# ---------------------------
# Publiczne API: process_pdf
# ---------------------------
def _process_extracted_text(
    pdf_text: str,
    offer_key: str,
    status_tracker: Optional[StatusTracker],
) -> List[Dict[str, Any]]:

    if pdf_text is None or not str(pdf_text).strip():
        raise ValueError("pdf_text is empty")

    if status_tracker is None:
        status_tracker = StatusTracker()

    scan_data = _phase1_scan_pdf(pdf_text, offer_key, status_tracker)

    total_items = len(scan_data.get("items_preview", []))

    if total_items == 0:
        print("Warning: No items found in Phase 1")
        return [_build_empty_offer(scan_data)]

    all_processed_items, failed_items = _phase2_batch_process_items(
        pdf_text, offer_key, scan_data["items_preview"], total_items, status_tracker
    )


    final_offer = _phase3_assemble_and_validate(
        pdf_text, scan_data, all_processed_items, failed_items, total_items, status_tracker
    )

    print("Final offer: ", final_offer)
    return [final_offer]


def process_pdf(
    input_file: bytes,
    status_tracker: Optional[StatusTracker] = None,
    *,
    offer_key: Optional[str] = None,
    text_cfg: Optional[PdfToTextConfig] = None,
    ocr: Optional[OcrEngine] = None,
) -> List[Dict[str, Any]]:
    if not input_file:
        raise ValueError("Input file is empty")

    print(f"Input file size: {len(input_file)} bytes")

    cfg = text_cfg or PdfToTextConfig()
    pdf_text = pdf_bytes_to_text(input_file, cfg=cfg, ocr=ocr)

    if offer_key is None:
        offer_key = "bytes_input"

    # opcjonalnie: zapis pdf_text do artifacts
    if SAVE_PHASE_ARTIFACTS:
        odir = _offer_dir(offer_key)
        _write_text(odir / "pdf_text.txt", pdf_text)

    return _process_extracted_text(pdf_text, offer_key, status_tracker)



def process_pdf_text(
    pdf_text: str,
    offer_key: str,
    status_tracker: Optional[StatusTracker] = None,
) -> List[Dict[str, Any]]:
    return _process_extracted_text(pdf_text, offer_key, status_tracker)



# ---------------------------
# Phase 1
# ---------------------------

def _phase1_scan_pdf(pdf_text: str, offer_key: str, status_tracker: StatusTracker) -> Dict[str, Any]:
    _update_progress(status_tracker, 5, "Faza 1: Skanowanie listy pozycji z tabel...", 1)
    print("PHASE 1: Initial Scan - Items Preview From Tables (LLM)")

    base_prompt = get_base_prompt()
    phase1_prompt = get_phase1_prompt(base_prompt)

    full_prompt = (
        phase1_prompt
        + "\n\n[PDF_TEXT_START]\n"
        + pdf_text
        + "\n[PDF_TEXT_END]\n"
    )

    # --- DODANE: zapis promptu Phase1 ---
    if SAVE_PHASE_ARTIFACTS:
        odir = _offer_dir(offer_key)
        _write_text(odir / "phase1.prompt.txt", full_prompt)

    try:
        scan_dict = call_llama_structured(
            full_prompt,
            OfferScan,
            max_new_tokens=2500,
        )

        # --- DODANE: zapis PRED Phase1 ---
        if SAVE_PHASE_ARTIFACTS:
            odir = _offer_dir(offer_key)
            _write_json(odir / "phase1.pred.json", scan_dict)
            meta = {
            "offer_key": offer_key,
            "phase": "phase1",
            "prompt_version": PROMPT_VERSION,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            "items_preview_count": len(scan_dict.get("items_preview", []) or []),
            }
            _write_json(odir / "phase1.meta.json", meta)

        items = scan_dict.get("items_preview", [])
        print(f"Found {len(items)} items in Phase 1 preview")

        scan_dict.setdefault("id", "")
        scan_dict.setdefault("nazwa_dostawcy", "")
        scan_dict.setdefault("data_oferty", None)

        return scan_dict

    except Exception as e:
        print(f"Error in Phase 1 (LLM): {e}")
        print("✗ Model nie zwrócił poprawnego JSON. Zwracam pustą strukturę oferty.")

        # --- DODANE: zapis info o błędzie (opcjonalnie) ---
        if SAVE_PHASE_ARTIFACTS:
            odir = _offer_dir(offer_key)
            _write_json(odir / "phase1.error.json", {"error": str(e)})

        return {
            "id": "",
            "nazwa_dostawcy": "",
            "data_oferty": None,
            "items_preview": [],
        }



# ---------------------------
# Phase 2
# ---------------------------

def _phase2_batch_process_items(
    pdf_text: str,
    offer_key: str,
    items_preview: List[Dict],
    total_items: int,
    status_tracker: StatusTracker,
) -> Tuple[List[Dict], List[Dict]]:
    print("PHASE 2: Batch Processing - Detailed Item Extraction")


    num_batches = (total_items + BATCH_SIZE - 1) // BATCH_SIZE
    all_processed_items: List[Dict[str, Any]] = []
    failed_items: List[Dict[str, Any]] = []

    progress_start = 15
    progress_end = 40
    progress_range = progress_end - progress_start

    for batch_idx in range(num_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_items)
        batch_items = items_preview[batch_start:batch_end]

        current_progress = progress_start + int(((batch_idx + 1) / num_batches) * progress_range)
        _update_progress(
            status_tracker,
            current_progress,
            f"Przetwarzanie partii {batch_idx + 1}/{num_batches} ({batch_end}/{total_items} pozycji)...",
            1,
        )

        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}: items {batch_start + 1} to {batch_end}")

        batch_results, batch_failed = _process_single_batch(
            pdf_text, offer_key, batch_items, batch_start, batch_idx, num_batches
        )


        all_processed_items.extend(batch_results)
        failed_items.extend(batch_failed)

    return all_processed_items, failed_items


def _process_single_batch(
    pdf_text: str,
    offer_key: str,
    batch_items: List[Dict],
    batch_start: int,
    batch_idx: int,
    num_batches: int,
) -> Tuple[List[Dict], List[Dict]]:
    base_prompt = get_base_prompt()

    def try_process_items(items: List[Dict], start_idx: int) -> List[Dict]:
        batch_prompt = get_phase2_prompt(base_prompt, items, start_idx)

        full_prompt = (
            batch_prompt
            + "\n\n[PDF_TEXT_START]\n"
            + pdf_text
            + "\n[PDF_TEXT_END]\n"
        )

        # --- zapis promptu Phase2 ---
        if SAVE_PHASE_ARTIFACTS:
            odir = _offer_dir(offer_key)
            batch_tag = f"{batch_idx:03d}"
            _write_text(odir / f"phase2.batch_{batch_tag}.prompt.txt", full_prompt)

        raw_results = call_llama_structured(full_prompt, list[OfferItem])

        # --- zapis PRED Phase2 ---
        if SAVE_PHASE_ARTIFACTS:
            odir = _offer_dir(offer_key)
            batch_tag = f"{batch_idx:03d}"
            _write_json(odir / f"phase2.batch_{batch_tag}.pred.json", raw_results)
            meta = {
            "offer_key": offer_key,
            "phase": "phase2",
            "batch_index": batch_idx,
            "batch_start": batch_start,
            "items_in_batch": len(batch_items),
            "prompt_version": PROMPT_VERSION,
            "created_utc": datetime.utcnow().isoformat() + "Z",
            }
            _write_json(odir / f"phase2.batch_{batch_tag}.meta.json", meta)

        # walidacje
        if not isinstance(raw_results, list):
            raise ValueError(f"Expected list[OfferItem], got: {type(raw_results)}")

        if len(raw_results) != len(items):
            raise ValueError(
                f"Cardinality mismatch: expected {len(items)} items, got {len(raw_results)}"
            )

        # normalizacja do list[dict]
        normalized: List[Dict[str, Any]] = []
        for item in raw_results:
            if isinstance(item, OfferItem):
                normalized.append(item.model_dump())
            else:
                normalized.append(item)

        return normalized

    # 1) całość batcha
    try:
        results = try_process_items(batch_items, batch_start)
        print(f"✓ Batch {batch_idx + 1} complete: {len(results)} items processed")
        return results, []
    except Exception as e:
        print(f"⚠ Full batch failed for batch {batch_idx + 1}: {e}")

        # opcjonalnie: log błędu Phase2 na dysk (pomaga debugować)
        if SAVE_PHASE_ARTIFACTS:
            odir = _offer_dir(offer_key)
            batch_tag = f"{batch_idx:03d}"
            _write_json(odir / f"phase2.batch_{batch_tag}.error.json", {"error": str(e)})

    # 2) połówki
    if len(batch_items) > 1:
        mid = len(batch_items) // 2
        results: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []

        halves = [
            (batch_items[:mid], batch_start),
            (batch_items[mid:], batch_start + mid),
        ]

        for half_idx, (half_items, half_start) in enumerate(halves, start=1):
            try:
                half_results = try_process_items(half_items, half_start)
                results.extend(half_results)
                print(f"  ✓ Half batch {half_idx}/2 complete: {len(half_results)} items")
            except Exception as half_err:
                print(f"  ⚠ Half batch {half_idx}/2 failed: {half_err}")
                print("  → Trying items individually...")

                for item_local_idx, item in enumerate(half_items):
                    item_global_pos = half_start + item_local_idx

                    for attempt in range(3):
                        try:
                            single_result = try_process_items([item], item_global_pos)
                            results.extend(single_result)
                            print(f"    ✓ Item {item_global_pos + 1} processed")
                            break
                        except Exception as single_err:
                            if attempt == 2:
                                print(f"    ✗ Item {item_global_pos + 1} failed after 3 attempts: {single_err}")
                                failed_items.append(item)
                            else:
                                print(f"    ⚠ Item {item_global_pos + 1} attempt {attempt + 1} failed: {single_err}")

        return results, failed_items

    # 3) jeden element: 3 próby
    for attempt in range(3):
        try:
            results = try_process_items(batch_items, batch_start)
            print(f"✓ Single item {batch_start + 1} processed")
            return results, []
        except Exception as e:
            if attempt == 2:
                print(f"✗ Single item {batch_start + 1} failed after 3 attempts: {e}")
                return [], batch_items
            print(f"⚠ Single item {batch_start + 1} attempt {attempt + 1} failed: {e}")

    return [], batch_items

# ---------------------------
# Phase 3
# ---------------------------

SERVICE_LINE_RE = re.compile(
    r"(?i)\b("
    r"koszt\s+(spedycji|transportu|dostawy)|spedycj|transport|przesyłk|"
    r"kurier|ups|gls|dhl|handling|opłata\s+manipulacyjna|"
    r"serwis|robocizna|dojazd|montaż|uruchomienie|instalac"
    r")\b"
)

def _is_service_line(opis: str) -> bool:
    if not opis:
        return False
    return bool(SERVICE_LINE_RE.search(str(opis)))




def _phase3_assemble_and_validate(
    pdf_text: str,
    scan_data: Dict[str, Any],
    all_processed_items: List[Dict[str, Any]],
    failed_items: List[Dict[str, Any]],
    total_items: int,
    status_tracker: StatusTracker,
) -> Dict[str, Any]:
    print("PHASE 3: Assembly & Validation")

    _update_progress(status_tracker, 50, "Walidacja i składanie wyników...", 1)

    raw_processed_count = len(all_processed_items)
    print(f"Total items processed (raw): {raw_processed_count}/{total_items}")

    base_offer_data = {
        k: v for k, v in scan_data.items() if k not in ["items_preview", "pozycje_oferty"]
    }

    offer_items = _deduplicate_items(list(all_processed_items or []))
    before_service = len(offer_items)
    offer_items = [it for it in offer_items if not _is_service_line(it.get("opis", ""))]

    if len(offer_items) != total_items:
        print(
            f"[INFO] Items count differs from Phase1 preview: "
            f"{len(offer_items)} vs {total_items} (dedup={before_service}, service_filtered={before_service - len(offer_items)})"
        )


    print(f"Total items after deduplication: {len(offer_items)}/{total_items}")

    for idx, item in enumerate(offer_items, start=1):
        item["id"] = str(idx)

    offer_data: Dict[str, Any] = dict(base_offer_data)
    offer_data["pozycje_oferty"] = offer_items

    # 1) ID oferty (nagłówek) – preferuj to co jest w scan_data, a jeśli puste, wyciągnij z pdf_text
    # UWAGA: scan_data["id"] w Phase1 bywa puste; tu naprawiamy deterministycznie.
    if not str(offer_data.get("id", "") or "").strip():
        offer_data["id"] = _extract_offer_id_from_pdf_text(pdf_text)

    # 2) Propagacja id_oferty do wszystkich pozycji (zachowujemy format z PDF!)
    offer_id_for_items = ""

    # a) spróbuj znaleźć pierwsze niepuste id_oferty zwrócone przez LLM
    for it in offer_items:
        candidate = str(it.get("id_oferty", "") or "").strip()
        if candidate:
            offer_id_for_items = candidate
            break

    # b) jeśli brak w itemach, weź z nagłówka
    if not offer_id_for_items:
        offer_id_for_items = str(offer_data.get("id", "") or "").strip()

    # c) jedna pętla: id_oferty + producent + business logic
    for item in offer_items:
        if not str(item.get("id_oferty", "") or "").strip():
            item["id_oferty"] = offer_id_for_items

        if item.get("producent") == "-":
            item["producent"] = ""

        _apply_business_logic_to_item(item)

    # liczba_elementow
    offer_data["liczba_elementow"] = len(offer_items)

    print("\n✓ Processing complete:")
    print(f"  - Offer ID: {offer_data.get('id', '')}")
    print(f"  - Supplier: {offer_data.get('nazwa_dostawcy', '')}")
    print(f"  - Items: {len(offer_items)}/{total_items}")
    print(f"  - Failed: {len(failed_items)}")

    return offer_data

NOISE_SERIAL_RE = re.compile(
    r"(?i)\b(pln|eur|vat|rabat|cena|wartość|ilość|termin|na stanie|https?://|www\.)\b"
)

def _looks_like_product_code(token: str) -> bool:
    """
    Do swapu serial->oem: krótki kod, zwykle alnum lub cyfry; bez słów i bez walut/linków.
    """
    if not token:
        return False
    t = token.strip()
    if NOISE_SERIAL_RE.search(t):
        return False
    # odrzuć ewidentne jednostki
    if t.lower() in {"szt", "szt.", "jm", "j.m.", "kg", "g", "m"}:
        return False
    # minimalna heurystyka: ma cyfry albo jest alfanumeryczny z myślnikiem/slashem
    has_digit = any(c.isdigit() for c in t)
    has_alpha = any(c.isalpha() for c in t)
    if has_alpha and has_digit:
        return True
    if has_digit and len(t) >= 4 and len(t) <= 20:
        return True
    if re.match(r"^[A-Z0-9][A-Z0-9\-\/\.]{2,30}$", t, re.IGNORECASE):
        return True
    return False


def _apply_business_logic_to_item(item: Dict[str, Any]) -> None:
    # 1) NIE normalizuj id_oferty do cyfr – ma zostać w formie z PDF
    if item.get("id_oferty") is None:
        item["id_oferty"] = ""

    opis = str(item.get("opis", "")).strip()
    numer_oem = str(item.get("numer_oem", "")).strip()
    serial_raw = str(item.get("serial_numbers", "")).strip()

    # 2) Wyczyść serial_numbers ze śmieci (ceny/rabaty/linki/terminy)
    if serial_raw and NOISE_SERIAL_RE.search(serial_raw):
        # zachowaj tylko poprawne elementy typu CN:/PKWiU:/EAN:/ID:
        parts = [p.strip() for p in serial_raw.split(";") if p.strip()]
        keep = []
        for p in parts:
            if re.match(r"(?i)^(CN:\s*\S+|PKWiU:\s*\S+|EAN:\s*\S+|ID:\s*\S+)$", p):
                keep.append(p)
        serial_raw = ";".join(keep)

    # 3) Swap serial_numbers -> numer_oem, gdy numer_oem pusty, a serial wygląda jak kod produktu
    if not numer_oem and serial_raw:
        parts = [p.strip() for p in serial_raw.split(";") if p.strip()]
        if len(parts) == 1 and _looks_like_product_code(parts[0]) and not _looks_like_classification_code(parts[0]):
            numer_oem = parts[0]
            serial_raw = ""

    # 4) Jeżeli numer_oem wygląda jak klasyfikacja (CN/PKWiU), przenieś do serial_numbers
    extra_codes: List[str] = []
    if numer_oem and _looks_like_classification_code(numer_oem):
        extra_codes.append(numer_oem)
        numer_oem = ""

    # 5) Heurystyczny OEM z opisu tylko jeśli nadal pusty (to jest „best effort”, ale ostrożnie)
    if not numer_oem:
        guessed = _guess_oem_from_description(opis)
        if guessed and not _looks_like_classification_code(guessed):
            numer_oem = guessed

    # 6) Sklej seriale + dodaj ewentualne CN/PKWiU z opisu
    serial_parts: List[str] = []
    if serial_raw:
        serial_parts = [p.strip() for p in serial_raw.split(";") if p.strip()]

    for code in extra_codes:
        if code and code not in serial_parts:
            serial_parts.append(code)

    if opis:
        for cn in CN_REGEX.findall(opis):
            token = f"CN: {cn}"
            if token not in serial_parts:
                serial_parts.append(token)
        for pk in PKWIU_REGEX.findall(opis):
            token = f"PKWiU: {pk}"
            if token not in serial_parts:
                serial_parts.append(token)

    # 7) Final clean: usuń duplikaty i parametry techniczne
    numer_oem_clean = numer_oem.strip()
    cleaned_serials: List[str] = []
    seen = set()

    TECH_PATTERNS = [" V", "V ", " kW", "kW ", " KW", "Hz", " RPM", " obr/min"]

    for p in serial_parts:
        if not p:
            continue
        up = p.upper()
        if any(tp.strip().upper() in up for tp in TECH_PATTERNS):
            continue
        if numer_oem_clean and (p == numer_oem_clean or numer_oem_clean in p):
            continue
        if p not in seen:
            seen.add(p)
            cleaned_serials.append(p)

    item["numer_oem"] = numer_oem_clean
    item["serial_numbers"] = ";".join(cleaned_serials) if cleaned_serials else ""

# ---------------------------
# Deduplikacja
# ---------------------------

def _normalize_description_for_key(opis: str) -> str:
    if not opis:
        return ""
    text = str(opis).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text[:80]


def _deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_items: List[Dict[str, Any]] = []

    for item in items:
        opis = item.get("opis") or ""
        opis_norm = _normalize_description_for_key(opis)

        numer_oem = (item.get("numer_oem") or "").strip()
        if not numer_oem:
            numer_oem = _guess_oem_from_description(opis)
        numer_oem_norm = (numer_oem or "").strip().lower()

        key = (numer_oem_norm, opis_norm)
        if key in seen:
            continue

        seen.add(key)
        item["numer_oem"] = numer_oem
        unique_items.append(item)

    return unique_items


def _build_empty_offer(scan_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **{k: v for k, v in scan_data.items() if k != "items_preview"},
        "pozycje_oferty": [],
    }




def run_single_pdf(
    pdf_path: str | Path,
    output_json_path: str | Path,
    *,
    text_cfg: Optional[PdfToTextConfig] = None,
    ocr: Optional[OcrEngine] = None,
    save_preview_txt: bool = False,
    preview_txt_path: Optional[str | Path] = None,
) -> int:
    """
    SINGLE: przetwarza jeden PDF -> zapisuje JSON.
    Opcjonalnie zapisuje też preview TXT (przydatne do debugowania).
    """
    pdf_path = Path(pdf_path)
    output_json_path = Path(output_json_path)

    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Invalid PDF path: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()

    cfg = text_cfg or PdfToTextConfig()
    pdf_text = pdf_bytes_to_text(pdf_bytes, cfg=cfg, ocr=ocr)

    if save_preview_txt:
        if preview_txt_path is None:
            preview_txt_path = output_json_path.with_suffix(".txt")
        preview_txt_path = Path(preview_txt_path)
        preview_txt_path.parent.mkdir(parents=True, exist_ok=True)
        preview_txt_path.write_text(pdf_text, encoding="utf-8")

    offer_key = pdf_path.stem

    # opcjonalnie: zapis pdf_text do artifacts
    if SAVE_PHASE_ARTIFACTS:
        odir = _offer_dir(offer_key)
        _write_text(odir / "pdf_text.txt", pdf_text)

    status = StatusTracker()
    results = process_pdf_text(pdf_text, offer_key, status)


    save_json(output_json_path, results)
    return 0


def run_batch_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    view_text_dir: Optional[str | Path] = None,
    text_cfg: Optional[PdfToTextConfig] = None,
    ocr: Optional[OcrEngine] = None,
    only_missing: bool = False,
) -> int:
    """
    BATCH: iteruje po *.pdf w input_dir.
      - zapisuje JSON do output_dir/<stem>.json
      - opcjonalnie zapisuje preview TXT do view_text_dir/<stem>.txt

    only_missing=True:
      - pomija plik, jeśli docelowy JSON już istnieje (szybkie iteracje).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    view_path: Optional[Path] = None
    if view_text_dir is not None:
        view_path = Path(view_text_dir)
        view_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Brak plików PDF w katalogu: {input_dir}")
        return 0

    cfg = text_cfg or PdfToTextConfig()

    failed = 0
    for idx, pdf_path in enumerate(pdf_files, start=1):
        out_json = output_dir / f"{pdf_path.stem}.json"
        out_txt = (view_path / f"{pdf_path.stem}.txt") if view_path else None

        if only_missing and out_json.exists():
            print(f"[{idx}/{len(pdf_files)}] SKIP (exists): {pdf_path.name}")
            continue

        print("\n" + "=" * 20)
        print(f"[{idx}/{len(pdf_files)}] Przetwarzam: {pdf_path.name}")
        print("=" * 20)

        try:
            pdf_bytes = pdf_path.read_bytes()

            # 1) PDF -> tekst
            pdf_text = pdf_bytes_to_text(pdf_bytes, cfg=cfg, ocr=ocr)

            # 2) Preview TXT (opcjonalnie)
            if out_txt is not None:
                out_txt.write_text(pdf_text, encoding="utf-8")
                print(f"Podgląd tekstu zapisany do: {out_txt}")

            # 3) offer_key + artifacts
            offer_key = pdf_path.stem
            if SAVE_PHASE_ARTIFACTS:
                odir = _offer_dir(offer_key)
                _write_text(odir / "pdf_text.txt", pdf_text)

            # 4) LLM processing
            status = StatusTracker()
            results = process_pdf_text(pdf_text, offer_key, status)

            # 5) Zapis final JSON
            save_json(out_json, results)
            print(f"Wynik LLM zapisany do: {out_json}")


        except Exception as e:
            failed += 1
            print(f"[ERROR] {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "-" * 20)
    print(f"Batch processing finished. Failed files: {failed}/{len(pdf_files)}")
    print("-" * 20)
    return 1 if failed else 0
