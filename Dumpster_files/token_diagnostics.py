"""
Standalone tool for inspecting token usage of LLaMA prompts.

Usage:
    python token_diagnostics.py
    python token_diagnostics.py /path/to/offer.pdf

What it does:
    - loads a PDF file (default path is configurable),
    - extracts text exactly like processor._pdf_bytes_to_text,
    - builds the Phase 1 prompt:
        base_prompt + phase1_prompt + [PDF_TEXT_START] ... [PDF_TEXT_END],
    - tokenizes the final USER prompt with the same tokenizer family (LLaMA),
    - prints:
        * tokens in base+phase1 instructions,
        * tokens in PDF_TEXT,
        * total tokens in user prompt,
        * estimated available margin for model output.
"""

import sys
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from processor import _pdf_bytes_to_text
from prompts import get_base_prompt, get_phase1_prompt


# ======================================================================
# CONFIGURATION – dostosuj do swojego modelu
# ======================================================================

# Ścieżka domyślna do PDF (jeśli nie podasz nic w argumencie)
DEFAULT_PDF_INPUT_PATH = "/home/karol_zych/przetwarzanie_ofert_projekt/1_144_weidmuller.pdf"

# Nazwa / ścieżka modelu – musi być zgodna z MODEL_PATH w llm_client.py
MODEL_PATH = "llama"

# Szacowane okno kontekstu modelu (ile tokenów łącznie: wejście + wyjście)
# Jeśli używasz innego okna (np. 8192), zmień to tutaj.
MODEL_CONTEXT_WINDOW = 4096

# Domyślnie zakładamy, że na output chcesz przeznaczyć tyle tokenów
DEFAULT_DESIRED_MAX_NEW_TOKENS = 1200

# ======================================================================


def build_phase1_user_prompt(pdf_bytes: bytes) -> str:
    """
    Buduje dokładnie taki user_prompt, jaki trafia do call_llama_structured
    w Fazie 1 (bez system_prompt i chat-template).
    """
    base_prompt = get_base_prompt()
    phase1_prompt = get_phase1_prompt(base_prompt)
    pdf_text = _pdf_bytes_to_text(pdf_bytes)

    full_prompt = (
        phase1_prompt
        + "\n\n[PDF_TEXT_START]\n"
        + pdf_text
        + "\n[PDF_TEXT_END]\n"
    )

    return full_prompt


def analyse_tokens_for_pdf(pdf_path: Path, desired_max_new_tokens: int = DEFAULT_DESIRED_MAX_NEW_TOKENS) -> None:
    """
    Wczytuje PDF, buduje prompt Faz y1, liczy tokeny i wypisuje najważniejsze informacje.
    """
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: file is not a PDF: {pdf_path}")
        return

    print("=" * 80)
    print(f"TOKEN DIAGNOSTICS – PHASE 1 PROMPT")
    print(f"PDF file: {pdf_path}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model context window (assumed): {MODEL_CONTEXT_WINDOW} tokens")
    print("=" * 80)

    # 1. Wczytanie PDF
    try:
        pdf_bytes = pdf_path.read_bytes()
        print(f"PDF loaded successfully ({len(pdf_bytes)} bytes)")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    # 2. Budowa promptu
    user_prompt = build_phase1_user_prompt(pdf_bytes)

    # 3. Tokenizer (tylko tokenizer, bez ładowania samego modelu)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # 3a. Tokeny osobno dla instrukcji i PDF_TEXT
    # Dla uproszczenia dzielimy po znaczniku [PDF_TEXT_START]
    split_marker = "[PDF_TEXT_START]"
    if split_marker in user_prompt:
        before_pdf, after_pdf = user_prompt.split(split_marker, 1)
        # w after_pdf jest jeszcze [PDF_TEXT_END]; i tak liczymy całość jako "PDF section"
        instructions_tokens = len(tokenizer(before_pdf).input_ids)
        pdf_section_tokens = len(tokenizer(split_marker + after_pdf).input_ids)
    else:
        # fallback: traktujemy całość jako jeden blok
        instructions_tokens = len(tokenizer(user_prompt).input_ids)
        pdf_section_tokens = 0

    # 3b. Tokeny całego user_prompt
    total_user_tokens = len(tokenizer(user_prompt).input_ids)

    # 4. Szacunek marginesu na output
    remaining_for_output = MODEL_CONTEXT_WINDOW - total_user_tokens
    recommended_max_new_tokens = max(0, min(desired_max_new_tokens, remaining_for_output))

    # 5. Wypisanie wyników
    print("\n--- TOKEN BREAKDOWN ---")
    print(f"Tokens in instructions (base + phase1, WITHOUT pdf_text):   {instructions_tokens}")
    print(f"Tokens in PDF section (including markers):                  {pdf_section_tokens}")
    print(f"Total tokens in user prompt (Phase 1 user_prompt):          {total_user_tokens}")
    print(f"Assumed model context window:                               {MODEL_CONTEXT_WINDOW}")

    print("\n--- OUTPUT BUDGET ---")
    print(f"Desired max_new_tokens:                                     {desired_max_new_tokens}")
    print(f"Remaining tokens for output (context - input):              {remaining_for_output}")
    print(f"Recommended max_new_tokens (clipped):                       {recommended_max_new_tokens}")

    if remaining_for_output <= 0:
        print("\nWARNING: Prompt alone already exceeds or fully fills the context window!")
        print("You must shorten the PDF text (fewer pages, lower MAX_CHARS) or the instructions.")
    elif remaining_for_output < desired_max_new_tokens:
        print("\nNOTE: Desired max_new_tokens is larger than the remaining context.")
        print("Consider reducing max_new_tokens or cutting the PDF further.")

    print("\nDiagnostic completed.\n")


def main(argv: Optional[list[str]] = None) -> int:
    """
    Główny punkt wejścia:
        - jeśli podasz ścieżkę w argumencie → użyje jej,
        - jeśli nie → użyje DEFAULT_PDF_INPUT_PATH.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) >= 1:
        pdf_path = Path(argv[0])
    else:
        pdf_path = Path(DEFAULT_PDF_INPUT_PATH)
        print(f"No PDF path provided, using default: {pdf_path}")

    analyse_tokens_for_pdf(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
