# pdf_to_text.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol
import io
import re
from pathlib import Path

import pdfplumber


# ---------------------------
# Konfiguracja
# ---------------------------

@dataclass(frozen=True)
class PdfToTextConfig:
    # Limity bezpieczeństwa (żeby nie wysadzić promptów / RAM)
    max_pages: int = 8
    max_chars: int = 40_000

    # Tabele: heurystyki jakości
    max_table_columns: int = 15
    min_table_density: float = 0.30

    # NOWE: odfiltruj mikrotabelki z nagłówka/stopki
    min_table_rows: int = 2
    min_table_cols: int = 4

    # Deduplikacja tabel vs tekst strony (jeśli tabela w dużym stopniu jest już w tekście)
    table_duplicate_line_ratio: float = 0.60
    min_line_len_for_duplicate_check: int = 10

    # Czyszczenie
    collapse_spaces: bool = True
    max_consecutive_newlines: int = 2

    # Czy dołączać nagłówki stron/sekcji (pomaga debugować)
    include_page_headers: bool = True



# ---------------------------
# OCR: interfejs (opcjonalny)
# ---------------------------

class OcrEngine(Protocol):
    """
    Interfejs OCR. Implementacja może bazować np. na:
    - Tesseract (pytesseract)
    - Google Document AI / Vision
    - Azure Form Recognizer / Read
    - PaddleOCR
    """
    def ocr_page_image_to_text(self, image_bytes: bytes, *, page_number: int) -> str:
        ...


# ---------------------------
# Narzędzia: czyszczenie i formatowanie
# ---------------------------

_whitespace_re = re.compile(r"[ \t]{2,}")
_newlines_re = re.compile(r"\n{3,}")

def clean_pdf_text(text: str, cfg: PdfToTextConfig) -> str:
    """Stabilizuje tekst z PDF (przydatne przed dalszym przetwarzaniem)."""
    if not text:
        return ""

    # twarde spacje
    text = text.replace("\xa0", " ")

    if cfg.collapse_spaces:
        text = _whitespace_re.sub(" ", text)

    if cfg.max_consecutive_newlines >= 1:
        # zamień 3+ nowych linii na max (cfg.max_consecutive_newlines)
        max_nl = "\n" * cfg.max_consecutive_newlines
        text = _newlines_re.sub(max_nl, text)

    return text.strip()


def format_table_as_text(table: List[List[Optional[str]]]) -> str:
    """
    Zamienia tabelę (list[list[str]]) na tekst.
    Każdy wiersz = jedna linia, kolumny rozdzielone ' | '.
    """
    lines: List[str] = []

    for row in table or []:
        if not row:
            continue
        cleaned_cells: List[str] = []
        for cell in row:
            c = cell or ""
            c = " ".join(c.split())  # usuwa \n, taby, wielokrotne spacje
            cleaned_cells.append(c)

        if not any(cleaned_cells):
            continue

        lines.append(" | ".join(cleaned_cells))

    return "\n".join(lines).strip()

LINE_ITEMS_HEADER_RE = re.compile(
    r"(?i)\b(indeks)\b.*\b(ilo(?:ść|sc))\b.*\b(jm|j\.m\.)\b.*\b(cena)\b.*\b(warto(?:ść|sc))\b"
)

LINE_ITEM_ROW_RE = re.compile(
    r"^\s*(\d{1,3})\s+(\S+)\s+(\d+(?:[.,]\d+)?)\s+(\S+)\s+"
    r"(\d+(?:[.,]\d+)?)\s+(PLN|EUR)\s+"
    r"(\d+(?:[.,]\d+)?)\s*%\s+"
    r"(\d+(?:[.,]\d+)?)\s+(PLN|EUR)\s+"
    r"(\d+(?:[.,]\d+)?)\s+(PLN|EUR)\s*$"
)

TOTALS_STOP_RE = re.compile(r"(?i)^\s*(razem|suma|podsumowanie)\b")

def synthesize_line_items_table_from_text(page_text: str) -> str:
    """
    Fallback: gdy pdfplumber nie wykrył tabeli pozycji,
    spróbuj zbudować ją z tekstu (nagłówek + wiersze).
    """
    if not page_text:
        return ""

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # znajdź nagłówek tabeli
    header_idx = None
    for i, ln in enumerate(lines):
        if LINE_ITEMS_HEADER_RE.search(ln):
            header_idx = i
            break
    if header_idx is None:
        return ""

    rows = []
    i = header_idx + 1

    current = None  # trzymamy bieżący rekord, żeby dołączyć opis z kolejnej linii
    while i < len(lines):
        ln = lines[i]

        if TOTALS_STOP_RE.search(ln):
            break

        m = LINE_ITEM_ROW_RE.match(ln)
        if m:
            # zamknij poprzedni
            if current is not None:
                rows.append(current)

            current = {
                "lp": m.group(1),
                "indeks": m.group(2),
                "ilosc": m.group(3),
                "jm": m.group(4),
                "cena_netto": m.group(5),
                "waluta1": m.group(6),
                "rabat_proc": m.group(7),
                "cena_po_rabacie": m.group(8),
                "waluta2": m.group(9),
                "wartosc_netto": m.group(10),
                "waluta3": m.group(11),
                "opis_i_termin": "",
            }
            i += 1
            continue

        # jeśli to nie jest nowy wiersz, a mamy current: doklej jako opis/termin
        if current is not None:
            # stopka/numery stron często zaczynają się od "Oferta nr", "strona"
            if re.search(r"(?i)\b(oferta\s*nr|strona)\b", ln):
                break
            if current["opis_i_termin"]:
                current["opis_i_termin"] += " " + ln
            else:
                current["opis_i_termin"] = ln

        i += 1

    if current is not None:
        rows.append(current)

    if not rows:
        return ""

    # zbuduj “tabelę” tekstową (kolumny stałe)
    out_lines = []
    out_lines.append("LP | Indeks | Ilość | JM | Cena netto | Rabat % | Cena po rabacie | Wartość netto | Opis / termin")
    for r in rows:
        out_lines.append(
            f"{r['lp']} | {r['indeks']} | {r['ilosc']} | {r['jm']} | "
            f"{r['cena_netto']} {r['waluta1']} | {r['rabat_proc']} | "
            f"{r['cena_po_rabacie']} {r['waluta2']} | {r['wartosc_netto']} {r['waluta3']} | "
            f"{r['opis_i_termin']}".strip()
        )

    return "\n".join(out_lines).strip()


def is_useful_table(table: List[List[Optional[str]]], cfg: PdfToTextConfig) -> bool:
    """
    Heurystyka: czy tabela jest warta dołączenia do tekstu.

    - odrzucamy bardzo szerokie (np. 30 kolumn),
    - odrzucamy bardzo puste (gęstość niepustych komórek < min_table_density).
    """
    if not table:
        return False

    min_rows = getattr(cfg, "min_table_rows", 2)
    min_cols = getattr(cfg, "min_table_cols", 4)

    n_rows = len([r for r in table if r and any((c or "").strip() for c in r)])
    if n_rows < min_rows:
        return False

    n_cols = 0
    for row in table:
        if row is not None:
            n_cols = max(n_cols, len(row))

    if n_cols < min_cols:
        return False
    if n_cols > cfg.max_table_columns:
        return False

    total_cells = 0
    nonempty_cells = 0
    for row in table:
        if row is None:
            continue
        for cell in row:
            total_cells += 1
            if cell and str(cell).strip():
                nonempty_cells += 1

    if total_cells == 0:
        return False

    density = nonempty_cells / total_cells
    return density >= cfg.min_table_density


_punct_strip_re = re.compile(r"[^\w\s\.\,/%:-]+", re.UNICODE)

def _normalize_for_duplicate_check(text: str) -> str:
    t = text or ""
    # usuń separatory tabel i “ASCII-art”
    t = t.replace("|", " ")
    # wyczyść znaki, które psują dopasowanie, ale zostaw liczby, /, %, ., :, -
    t = _punct_strip_re.sub(" ", t)
    # ujednolicenie białych znaków
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t



def is_table_duplicate_of_page_text(
    table_text: str,
    page_text: str,
    cfg: PdfToTextConfig,
) -> bool:
    """
    Sprawdza, czy tabela jest w większości już obecna w tekście strony.
    """
    if not table_text or not page_text:
        return False

    norm_page = _normalize_for_duplicate_check(page_text)
    if not norm_page:
        return False

    lines = [ln.strip() for ln in table_text.splitlines() if ln.strip()]
    if not lines:
        return False

    dup_lines = 0
    checked = 0

    for ln in lines:
        norm_ln = _normalize_for_duplicate_check(ln)
        if len(norm_ln) <= cfg.min_line_len_for_duplicate_check:
            continue
        checked += 1
        if norm_ln in norm_page:
            dup_lines += 1

    if checked == 0:
        return False

    return (dup_lines / checked) > cfg.table_duplicate_line_ratio


# ---------------------------
# Główna funkcja: PDF bytes -> tekst
# ---------------------------

def pdf_bytes_to_text(
    pdf_bytes: bytes,
    cfg: Optional[PdfToTextConfig] = None,
    ocr: Optional[OcrEngine] = None,
) -> str:
    """
    Konwertuje PDF (bytes) do tekstu.

    Strategia:
    1) pdfplumber: extract_text()
    2) pdfplumber: extract_tables() -> format_table_as_text() (tylko użyteczne tabele)
       - z deduplikacją tabel względem tekstu strony
    3) OCR (opcjonalnie): jeśli strona ma bardzo mało tekstu, można dołożyć OCR (hook)
       - na razie tylko miejsce integracji; implementacja OCR zależy od biblioteki/usługi.

    Zwraca:
      - Jeden spójny string, obcięty do cfg.max_chars.
    """

    if not pdf_bytes:
        raise ValueError("pdf_bytes is empty")

    cfg = cfg or PdfToTextConfig()

    texts: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            if page_idx >= cfg.max_pages:
                break

            page_parts: List[str] = []

            # 1) Tekst ciągły
            raw_text = page.extract_text() or ""
            cleaned_text = clean_pdf_text(raw_text, cfg)

            if cfg.include_page_headers:
                if cleaned_text:
                    page_parts.append(f"=== PAGE {page_idx + 1} TEXT ===\n{cleaned_text}")
            else:
                if cleaned_text:
                    page_parts.append(cleaned_text)
                    
            # 1b) Fallback: syntetyczna tabela pozycji z tekstu, jeśli wygląda na ofertę z tabelą
            synth_table = synthesize_line_items_table_from_text(cleaned_text)
            if synth_table:
                if cfg.include_page_headers:
                    page_parts.append(f"=== PAGE {page_idx + 1} TABLE LINE_ITEMS (SYNTH) ===\n{synth_table}")
                else:
                    page_parts.append(synth_table)


            # 2) Tabele
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for t_idx, table in enumerate(tables):
                if not is_useful_table(table, cfg):
                    continue

                table_text = format_table_as_text(table)
                if not table_text:
                    continue

                # deduplikacja: pomiń tabele, które są głównie kopią tekstu strony
                if is_table_duplicate_of_page_text(table_text, cleaned_text, cfg):
                    continue

                if cfg.include_page_headers:
                    page_parts.append(
                        f"=== PAGE {page_idx + 1} TABLE {t_idx + 1} ===\n{table_text}"
                    )
                else:
                    page_parts.append(table_text)

            # 3) OCR hook (opcjonalnie)
            # W praktyce: jeżeli cleaned_text jest bardzo krótki, a PDF jest skanem,
            # to można zrenderować stronę do obrazu i puścić OCR.
            #
            # Uwaga: pdfplumber może renderować stronę do obrazu przez page.to_image(...)
            # ale docelową implementację OCR dodamy w osobnym module.
            if ocr is not None:
                # Minimalny warunek (do modyfikacji): mało tekstu => spróbuj OCR
                if len(cleaned_text) < 40:
                    try:
                        # Render do obrazu (PNG bytes)
                        pil_img = page.to_image(resolution=200).original
                        img_buf = io.BytesIO()
                        pil_img.save(img_buf, format="PNG")
                        ocr_text = ocr.ocr_page_image_to_text(
                            img_buf.getvalue(),
                            page_number=page_idx + 1
                        )
                        ocr_text = clean_pdf_text(ocr_text, cfg)
                        if ocr_text:
                            if cfg.include_page_headers:
                                page_parts.append(
                                    f"=== PAGE {page_idx + 1} OCR ===\n{ocr_text}"
                                )
                            else:
                                page_parts.append(ocr_text)
                    except Exception:
                        # OCR to opcja „best effort” – nie blokujemy całego procesu
                        pass

            if page_parts:
                texts.append("\n\n".join(page_parts))

    full_text = "\n\n".join(texts)
    if not full_text:
        return ""

    return full_text[: cfg.max_chars]


# ---------------------------
# Prosty wrapper „publiczny”
# ---------------------------

def extract_text_from_pdf_file(
    pdf_path: str,
    cfg: Optional[PdfToTextConfig] = None,
    ocr: Optional[OcrEngine] = None,
    *,
    encoding: str = "utf-8",
) -> str:
    """
    Czyta PDF z dysku i zwraca tekst.
    encoding nie jest używany (PDF to bytes), ale zostawiam jako miejsce na przyszłe logi.
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes_to_text(pdf_bytes, cfg=cfg, ocr=ocr)


class BatchOfferProcessor:
    """
    Batch: PDF -> TXT preview.
    Katalogi:
      - input_dir: oferty_pdf/
      - view_dir:  view_text/
    """

    def __init__(
        self,
        input_dir: str,
        view_dir: str,
        text_cfg: Optional[PdfToTextConfig] = None,
        ocr: Optional[OcrEngine] = None,
    ):
        self.input_dir = Path(input_dir)
        self.view_dir = Path(view_dir)
        self.view_dir.mkdir(parents=True, exist_ok=True)

        self.text_cfg = text_cfg or PdfToTextConfig()
        self.ocr = ocr

    def run(self):
        pdf_files = sorted(self.input_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"Brak plików PDF w katalogu: {self.input_dir}")
            return

        print(f"Znaleziono {len(pdf_files)} plików PDF do przetworzenia.\n")

        for pdf_path in pdf_files:
            self._process_single_file(pdf_path)

    def _process_single_file(self, pdf_path: Path):
        print("\n" + "=" * 80)
        print(f"Rozpoczynam przetwarzanie: {pdf_path.name}")
        print("=" * 80)

        try:
            pdf_bytes = pdf_path.read_bytes()
        except Exception as e:
            print(f"Nie można odczytać pliku {pdf_path}: {e}")
            return

        # 1) PDF -> TEXT
        try:
            pdf_text = pdf_bytes_to_text(pdf_bytes, cfg=self.text_cfg, ocr=self.ocr)
        except Exception as e:
            print(f"Błąd podczas ekstrakcji tekstu z PDF: {e}")
            return

        # 2) Zapis podglądu
        view_file = self.view_dir / (pdf_path.stem + ".txt")
        try:
            view_file.write_text(pdf_text, encoding="utf-8")
            print(f"Podgląd tekstu zapisany do: {view_file}")
        except Exception as e:
            print(f"Nie udało się zapisać pliku podglądowego: {e}")
            return

        print(f"✓ Zakończono ekstrakcję tekstu: {pdf_path.name}")


if __name__ == "__main__":
    processor = BatchOfferProcessor(
        input_dir="INPUT/oferty_pdf",
        view_dir="OUTPUT/view_text",
        text_cfg=PdfToTextConfig(max_pages=8, max_chars=40_000, include_page_headers=True),
        ocr=None,
    )
    processor.run()