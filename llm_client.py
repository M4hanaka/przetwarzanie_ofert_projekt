# llm_client.py
# --------------------------------------------------------------------------------------
# Klient LLM (Qwen3-8B) + Outlines (wymuszanie struktury JSON).
#
# Cel poprawek:
# - Automatyczne dopasowanie liczby tokenów wyjściowych tak, aby nie dostawać 32 tokenów,
#   co prowadzi do uciętego JSON ("Unterminated string ...").
# - Gdy prompt jest zbyt długi, ucinamy WYŁĄCZNIE fragment PDF_TEXT pomiędzy markerami
#   [PDF_TEXT_START] i [PDF_TEXT_END], rezerwując stabilny budżet na wyjście JSON.
#
# Założenia:
# - Nie stosujemy strategii "bierz końcówkę PDF" (Twoje wymaganie).
# - Zachowujemy Qwen chat template i wyłączamy thinking.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, Optional, Type

import torch
import outlines
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------------------------------------------
# Konfiguracja modelu
# --------------------------------------------------------------------------------------

MODEL_PATH: str = "/home/karol_zych/MODELS/Qwen3-8B"

# Maksymalne okno kontekstu (input + output) w tokenach.
MAX_CONTEXT_TOKENS: int = 30_000

# Domyślna górna granica dla generacji (docelowa); faktycznie i tak przycinamy dynamicznie.
DEFAULT_MAX_NEW_TOKENS: int = 10_000

# Minimalny budżet tokenów na wyjście w trybie ekstrakcji (żeby JSON miał szansę się domknąć).
# Dla listy OfferItem z opisami sensowne są wartości 1200–2000.
MIN_OUTPUT_TOKENS: int = 1200

# Zapas na tokeny specjalne / narzut template / drobne różnice w tokenizacji.
SAFETY_MARGIN_TOKENS: int = 128


# --------------------------------------------------------------------------------------
# Ładowanie modelu
# --------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)

# Outlines: generator z dekodowaniem wymuszającym strukturę.
outlines_model = outlines.from_transformers(model, tokenizer)


SYSTEM_PROMPT: str = """
You extract structured information from PDF offer text.
Focus on meaning. The JSON formatting is handled by the decoder.
""".strip()


# --------------------------------------------------------------------------------------
# Narzędzia pomocnicze
# --------------------------------------------------------------------------------------

def _build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Buduje prompt w formacie chat dla Qwen3-8B.
    Używa apply_chat_template z wyłączonym 'thinking mode',
    żeby nie generować <think>...</think>, które mogłoby zniszczyć JSON.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt_text
    except TypeError:
        # Starsza wersja transformers bez parametru enable_thinking
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt_text
        except Exception as exception:
            print("Błąd _build_chat_prompt (fallback):", repr(exception))
            return ""
    except Exception as exception:
        print("Błąd _build_chat_prompt:", repr(exception))
        return ""


def _count_tokens(text: str) -> int:
    """Liczy tokeny w tekście przy użyciu tokenizera modelu."""
    if not text:
        return 0
    return len(tokenizer(text).input_ids)


def _truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    """
    Deterministycznie ucina tekst do max_tokens tokenów.
    Uwaga: zgodnie z wymaganiem NIE bierzemy końcówki, tylko początek.
    """
    if not text:
        return ""
    if max_tokens <= 0:
        return ""

    token_ids = tokenizer(text).input_ids
    if len(token_ids) <= max_tokens:
        return text

    truncated_token_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_token_ids, skip_special_tokens=True)


def _split_user_prompt_pdf_block(user_prompt: str) -> Optional[Tuple[str, str, str]]:
    """
    Rozpoznaje format:
        ... [PDF_TEXT_START]\n <pdf_text> \n[PDF_TEXT_END]\n ...

    Zwraca:
        (prefix_with_start_tag, pdf_text_between_tags, suffix_with_end_tag_and_rest)
    albo None jeśli markerów brak.
    """
    start_tag = "[PDF_TEXT_START]"
    end_tag = "[PDF_TEXT_END]"

    start_index = user_prompt.find(start_tag)
    if start_index == -1:
        return None

    # Szukaj END dopiero po START (a nie od początku całego promptu)
    end_index = user_prompt.find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return None

    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return None

    prefix = user_prompt[: start_index + len(start_tag)]
    middle = user_prompt[start_index + len(start_tag) : end_index]
    suffix = user_prompt[end_index:]  # zawiera end_tag i resztę promptu

    # Stabilizacja: usuwamy pojedynczą nową linię na początku/końcu, jeśli występuje.
    if middle.startswith("\n"):
        middle = middle[1:]
    if middle.endswith("\n"):
        middle = middle[:-1]

    return prefix, middle, suffix


def _build_prompt_with_dynamic_budget(
    system_prompt: str,
    user_prompt: str,
    requested_max_new_tokens: Optional[int],
    *,
    context_token_limit: int = MAX_CONTEXT_TOKENS,
    min_output_tokens: int = MIN_OUTPUT_TOKENS,
    safety_margin_tokens: int = SAFETY_MARGIN_TOKENS,
) -> Tuple[str, int]:
    """
    Buduje finalny prompt i dobiera max_new_tokens w sposób odporny na przepełnienie kontekstu.

    Strategia:
    1) Rezerwujemy stabilny budżet na output (co najmniej min_output_tokens).
    2) Jeśli prompt jest za długi, to ucinamy WYŁĄCZNIE pdf_text między markerami.
    3) Dopiero na końcu wyliczamy efektywny max_new_tokens jako min(output_budget, remaining).

    Zwraca:
        (prompt_text_ready_for_generation, effective_max_new_tokens)
    """
    # Ustalenie docelowego budżetu na output:
    if requested_max_new_tokens is None:
        requested_max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    # Chcemy co najmniej min_output_tokens, ale nie więcej niż requested_max_new_tokens.
    output_token_budget = max(min_output_tokens, min(requested_max_new_tokens, DEFAULT_MAX_NEW_TOKENS))

    pdf_parts = _split_user_prompt_pdf_block(user_prompt)

    # Jeśli nie ma markerów PDF_TEXT_START/END, nie mamy co ucinać selektywnie.
    if pdf_parts is None:
        prompt_text = _build_chat_prompt(system_prompt, user_prompt)
        input_tokens = _count_tokens(prompt_text)

        remaining_tokens = context_token_limit - input_tokens - safety_margin_tokens
        # Nigdy nie schodzimy do absurdalnie małej wartości, bo JSON się nie domknie.
        effective_max_new_tokens = max(256, min(output_token_budget, remaining_tokens))

        print(
            f"[LLM] input_tokens={input_tokens} "
            f"remaining_tokens={remaining_tokens} "
            f"effective_max_new_tokens={effective_max_new_tokens}"
        )

        return prompt_text, effective_max_new_tokens

    prefix_with_start_tag, pdf_text, suffix_with_end_tag = pdf_parts

    # Najpierw policz koszt bazowy promptu bez PDF (pdf_text jako pusty).
    user_prompt_without_pdf = prefix_with_start_tag + "\n" + suffix_with_end_tag
    base_prompt_text = _build_chat_prompt(system_prompt, user_prompt_without_pdf)
    base_prompt_tokens = _count_tokens(base_prompt_text)

    # Obliczamy, ile tokenów możemy przeznaczyć na pdf_text,
    # uwzględniając rezerwę na output i margines bezpieczeństwa.
    available_tokens_for_pdf_text = (
        context_token_limit
        - base_prompt_tokens
        - output_token_budget
        - safety_margin_tokens
    )

    # Minimalnie zostawiamy trochę miejsca na PDF, nawet w sytuacji skrajnej.
    if available_tokens_for_pdf_text < 256:
        available_tokens_for_pdf_text = 256

    # Ucinamy PDF deterministycznie (od początku), aby zachować istotne początki dokumentu,
    # co jest ważne dla ofert gdzie tabela jest na stronie 1–2.
    truncated_pdf_text = _truncate_text_to_token_budget(pdf_text, available_tokens_for_pdf_text)

    rebuilt_user_prompt = (
        prefix_with_start_tag
        + "\n"
        + truncated_pdf_text
        + "\n"
        + suffix_with_end_tag
    )

    prompt_text = _build_chat_prompt(system_prompt, rebuilt_user_prompt)
    input_tokens = _count_tokens(prompt_text)

    remaining_tokens = context_token_limit - input_tokens - safety_margin_tokens

    # Dajemy modelowi sensowny budżet na wyjście:
    # - preferujemy output_token_budget,
    # - ale nie przekraczamy remaining_tokens.
    effective_max_new_tokens = max(256, min(output_token_budget, remaining_tokens))

    print(
        f"[LLM] base_prompt_tokens(no_pdf)={base_prompt_tokens} "
        f"available_tokens_for_pdf_text={available_tokens_for_pdf_text} "
        f"input_tokens={input_tokens} remaining_tokens={remaining_tokens} "
        f"effective_max_new_tokens={effective_max_new_tokens}"
    )
    print(f"[LLM] pdf_text_chars={len(pdf_text)}")
    print(f"[LLM] pdf_text_tokens_before_trunc={_count_tokens(pdf_text)}")

    return prompt_text, effective_max_new_tokens


# --------------------------------------------------------------------------------------
# Publiczne API
# --------------------------------------------------------------------------------------

def call_llama_structured(
    user_prompt: str,
    output_type: Type[BaseModel] | type,
    system_prompt: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
):
    """
    Wywołanie Qwen3-8B z wymuszeniem struktury JSON (Outlines).
    Zwraca:
      - dict (jeśli output_type to BaseModel),
      - list[dict] (jeśli output_type to np. list[OfferItem]),
      - lub czysty obiekt Pythona zgodny ze schematem.

    Kluczowe zabezpieczenie:
    - dynamiczne budżetowanie tokenów wejścia/wyjścia w _build_prompt_with_dynamic_budget().
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    prompt_text, effective_max_new_tokens = _build_prompt_with_dynamic_budget(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        requested_max_new_tokens=max_new_tokens,
        context_token_limit=MAX_CONTEXT_TOKENS,
        min_output_tokens=MIN_OUTPUT_TOKENS,
        safety_margin_tokens=SAFETY_MARGIN_TOKENS,
    )

    try:
        result = outlines_model(
            prompt_text,
            output_type,
            max_new_tokens=effective_max_new_tokens,
        )
    except Exception as exception:
        # Debug: jeśli walidacja Outlines padła, spróbuj zrzucić "raw" wynik jako str.
        print("\n[DEBUG] Outlines validation failed:", repr(exception))
        try:
            raw_text = outlines_model(
                prompt_text,
                str,
                max_new_tokens=effective_max_new_tokens,
            )
            print("[DEBUG] RAW OUTPUT (truncated):", repr(str(raw_text)[:1000]))
        except Exception as second_exception:
            print("[DEBUG] Failed to get raw output:", repr(second_exception))
        raise

    # 1) BaseModel -> dict
    if isinstance(result, BaseModel):
        return result.model_dump()

    # 2) Jeśli dostaniemy JSON jako string, próbujemy parsować.
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Ostateczny fallback dla BaseModel (jeśli output_type jest BaseModel).
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                obj = output_type.model_validate_json(result)
                return obj.model_dump()
            raise

    # 3) Lista BaseModel lub lista dict -> normalizacja do list[dict]
    if isinstance(result, list):
        normalized_list = []
        for element in result:
            if isinstance(element, BaseModel):
                normalized_list.append(element.model_dump())
            else:
                normalized_list.append(element)
        return normalized_list

    # 4) Inne przypadki – zwracamy jak jest.
    return result


def save_json(path: str | Path, data: Any) -> None:
    """Zapisuje dane jako JSON UTF-8."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
