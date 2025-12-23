"""
Train Qwen (CausalLM) with SFT + LoRA on JSONL chat records.

Cel skryptu:
- Wczytać rekordy czatu z plików JSONL (train/val), gdzie każdy rekord ma pole "messages".
- Zbudować poprawny tekst wejściowy zgodny z chat_template tokenizera Qwen.
- Przeprowadzić Supervised Fine-Tuning (SFT) z użyciem adapterów LoRA (PEFT),
  aby model nauczył się generować odpowiedź (assistant) w formacie poprawnego JSON.

Założenia dot. danych:
- train.jsonl / val.jsonl zawierają rekordy z polem "messages" w formacie:
  [{"role": "system|user|assistant", "content": ...}, ...]
- "content" musi być stringiem
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

BASE_MODEL_DIRECTORY_PATH = "/home/karol_zych/MODELS/Qwen"

TRAINING_JSONL_FILE_PATH = "INPUT_SFT/train_json/trainn.jsonl"
VALIDATION_JSONL_FILE_PATH = "INPUT_SFT/train_json/vall.jsonl"

LORA_ADAPTER_OUTPUT_DIRECTORY_PATH = "/home/karol_zych/MODELS/lora_adapter"

MAXIMUM_SEQUENCE_TOKENS = 15000  # 4092
TRAINING_BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4
NUMBER_OF_EPOCHS = 6.0
WARMUP_RATIO = 0.03

LOGGING_STEPS = 8
SAVE_STEPS = 100 #200
RANDOM_SEED = 42

LORA_RANK = 16 #wyższa wartość = większa pojemność, więcej pamięci
LORA_ALPHA = 32 #LORA_ALPHA=LORA_RANK*2
LORA_DROPOUT = 0.05

USE_GRADIENT_CHECKPOINTING = True


def convert_any_value_to_text(value: Any) -> str:
    """
    Ujednolica dowolny typ do stringa.
    Model i tokenizer pracują na TEKŚCIE, więc jeżeli w dataset trafia dict/list,
    zamieniamy go na tekst JSON (json.dumps).
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_chat_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Funkcja zapewnia, że content jest stringiem oraz ujednolica listę messages do formatu:
    [{"role": "system|user|assistant", "content": "<tekst>"}, ...]
    """
    normalized_messages: List[Dict[str, str]] = []

    for message in raw_messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Nieprawidłowa rola w messages: {role!r}")

        normalized_messages.append(
            {
                "role": role,
                "content": convert_any_value_to_text(content),
            }
        )
    return normalized_messages


def build_training_text_from_messages(tokenizer, raw_messages: List[Dict[str, Any]]) -> str:
    #Buduje finalny tekst wejściowy dla modelu w formacie wymaganym przez Qwen chat_template.

    normalized_messages = normalize_chat_messages(raw_messages)
    apply_template_parameters = dict(tokenize=False, add_generation_prompt=False)

    try:
        return tokenizer.apply_chat_template(
            normalized_messages,
            enable_thinking=False,
            **apply_template_parameters,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            normalized_messages,
            **apply_template_parameters,
        )


def select_lora_target_module_names(model) -> List[str]:
    """
    Zwraca nazwy modułów, w które wstrzykniemy adaptery LoRA.
    Standardowo dla architektur Qwen/LLaMA:
    - Attention: q_proj, k_proj, v_proj, o_proj
    - MLP: gate_proj, up_proj, down_proj
    """
    preferred_target_module_names = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    all_module_names = [module_name for module_name, _ in model.named_modules()]

    contains_any_preferred_module = any(
        any(preferred_name in module_name for preferred_name in preferred_target_module_names)
        for module_name in all_module_names
    )

    if contains_any_preferred_module:
        return preferred_target_module_names

    raise RuntimeError("Nie znaleziono standardowych nazw modułów LoRA.\n")

#Callback do logowania użycia pamięci GPU (VRAM) podczas treningu.
class CUDAMemoryUsageCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return

        allocated_gb = torch.cuda.memory_allocated() / 1024**3 #realnie zajęte przez tensory
        reserved_gb = torch.cuda.memory_reserved() / 1024**3 #zarezerwowane przez cache PyTorch
        max_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

        print(
            f"[VRAM] step={state.global_step} "
            f"allocated={allocated_gb:.2f}GB reserved={reserved_gb:.2f}GB "
            f"max_allocated={max_allocated_gb:.2f}GB max_reserved={max_reserved_gb:.2f}GB"
        )


def main() -> None:
    # 0) Walidacja ścieżek do plików
    training_file_path = Path(TRAINING_JSONL_FILE_PATH)
    validation_file_path = Path(VALIDATION_JSONL_FILE_PATH)

    if not training_file_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku treningowego JSONL: {training_file_path}")
    if not validation_file_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku walidacyjnego JSONL: {validation_file_path}")

    # 1) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIRECTORY_PATH, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = MAXIMUM_SEQUENCE_TOKENS

    # 2) Model bazowy
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIRECTORY_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if base_model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    if USE_GRADIENT_CHECKPOINTING:
        base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False

    # 3) Wczytanie datasetu
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(training_file_path),
            "validation": str(validation_file_path),
        },
    )

    # 4) Funkcja formatująca przykład -> tekst rozmowy w chat_template
    def formatting_function(example: Dict[str, Any]) -> str:
        raw_messages = example["messages"]
        return build_training_text_from_messages(tokenizer, raw_messages)

    # 5) Konfiguracja LoRA (PEFT)
    lora_target_module_names = select_lora_target_module_names(base_model)

    lora_configuration = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_module_names,
    )

    # 6) Konfiguracja treningu (SFTConfig)
    training_configuration = SFTConfig(
        output_dir=LORA_ADAPTER_OUTPUT_DIRECTORY_PATH,
        num_train_epochs=NUMBER_OF_EPOCHS,
        per_device_train_batch_size=TRAINING_BATCH_SIZE_PER_DEVICE,
        per_device_eval_batch_size=max(1, TRAINING_BATCH_SIZE_PER_DEVICE),
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to="none",
        seed=RANDOM_SEED,
        logging_first_step=True,
    )

    # 7) SFTTrainer
    trainer = SFTTrainer(
        model=base_model,
        args=training_configuration,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_configuration,
        formatting_func=formatting_function,
        callbacks=[CUDAMemoryUsageCallback()],
    )

    # 8) Trening
    trainer.train()

    # 9) Zapis
    output_directory_path = Path(LORA_ADAPTER_OUTPUT_DIRECTORY_PATH)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    trainer.save_model(LORA_ADAPTER_OUTPUT_DIRECTORY_PATH)
    tokenizer.save_pretrained(LORA_ADAPTER_OUTPUT_DIRECTORY_PATH)

    print(f"\n[OK] LoRA adapter zapisany do: {LORA_ADAPTER_OUTPUT_DIRECTORY_PATH}")


if __name__ == "__main__":
    main()