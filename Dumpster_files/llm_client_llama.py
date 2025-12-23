import torch 
import outlines 
from pydantic import BaseModel 
import json 
from transformers import ( AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, )
MODEL_PATH = "MODELS/llama"

# Maksymalna długość KONTEKSTU modelu (wejście + wyjście)
MAX_INPUT_TOKENS = 10000 #4096
# Domyślna wartość docelowa dla max_new_tokens (będzie przycinana dynamicznie)
MAX_NEW_TOKENS = 3000

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)

outlines_model = outlines.from_transformers(model, tokenizer)

SYSTEM_PROMPT = """
You extract structured information from PDF offer text.
Focus on meaning. The JSON formatting is handled by the decoder.
"""

def _build_llama_prompt(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception as e:
        return print("Error - _build_llama_prompt")



def _count_tokens(text: str) -> int:
    """Policz tokeny w tekście przy użyciu tokenizera modelu."""
    return len(tokenizer(text).input_ids)


def _compute_safe_max_new_tokens(
    prompt_text: str,
    desired_max_new_tokens: int | None = None,
    context_window: int = MAX_INPUT_TOKENS,
    safety_margin: int = 64,
) -> int:
    """
    Dobierz bezpieczny max_new_tokens:
    - nie przekraczamy context_window,
    - zostawiamy safety_margin tokenów zapasu,
    - przycinamy desired_max_new_tokens jeśli trzeba.
    """
    if desired_max_new_tokens is None:
        desired_max_new_tokens = MAX_NEW_TOKENS

    input_tokens = _count_tokens(prompt_text)
    remaining = context_window - input_tokens - safety_margin

    if remaining <= 0:
        # prompt już wciska się w kontekst – dajemy minimalny ogon na wyjście
        return 32

    # nigdy mniej niż 32 tokeny, ale nie więcej niż pozostaje w kontekście
    return max(32, min(desired_max_new_tokens, remaining))


def call_llama_structured(
    user_prompt: str,
    output_type: type[BaseModel] | type,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
):
    """
    Wywołanie LLaMA z gramatycznie wymuszonym JSON (Outlines).
    Zwraca:
      - dict (jeśli output_type to BaseModel),
      - list[dict] (jeśli output_type to np. list[OfferItem]),
      - lub czysty Python object zgodny ze schematem.
    """
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT

    prompt_text = _build_llama_prompt(system_prompt, user_prompt)
    effective_max_new_tokens = _compute_safe_max_new_tokens(
        prompt_text,
        desired_max_new_tokens=max_new_tokens,
    )

    try:
        result = outlines_model(
            prompt_text,
            output_type,
            max_new_tokens=effective_max_new_tokens,
        )
    except Exception as e:
        print("\n[DEBUG] Outlines validation failed:", repr(e))
        try:
            raw_text = outlines_model(
                prompt_text,
                str,
                max_new_tokens=effective_max_new_tokens,
            )
            print("[DEBUG] RAW OUTPUT (truncated):", repr(str(raw_text)[:1000]))
        except Exception as e2:
            print("[DEBUG] Failed to get raw output:", repr(e2))
        raise

    # 1) BaseModel -> dict
    if isinstance(result, BaseModel):
        return result.model_dump()

    # 2) String JSON – spróbujmy sparsować
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            if isinstance(output_type, type) and issubclass(output_type, BaseModel):
                obj = output_type.model_validate_json(result)
                return obj.model_dump()
            raise

    # 3) Lista BaseModel lub lista dict
    if isinstance(result, list):
        normalized = []
        for elem in result:
            if isinstance(elem, BaseModel):
                normalized.append(elem.model_dump())
            else:
                normalized.append(elem)
        return normalized

    # 4) Inny przypadek – zwracamy jak jest
    return result
