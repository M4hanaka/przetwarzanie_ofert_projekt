#!/usr/bin/env python
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union


GOLD_JSON_DIR = Path("gold_json")
LLM_OUTPUT_DIR = Path("LLM_output_Qwen")  # jeśli używasz LLM_output_Qwen, zmień tutaj


FIELDS_TO_COMPARE = ["opis", "grupa_materialow", "numer_oem", "producent", "serial_numbers"]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_offer_object(raw: Any, path: Path, label: str) -> Union[Dict[str, Any], None]:
    """
    Przyjmuje dowolny JSON (dict albo list) i próbuje wyciągnąć obiekt z 'pozycje_oferty'.

    Obsługiwane przypadki:
    - {"id": "...", "pozycje_oferty": [...]}
    - [{"id": "...", "pozycje_oferty": [...]}]  # lista ofert, bierzemy pierwszą
    """
    # Przypadek 1: już dict
    if isinstance(raw, dict):
        if "pozycje_oferty" in raw:
            return raw
        else:
            print(f"  [{label}] Uwaga: w pliku {path.name} dict nie ma pola 'pozycje_oferty'.")
            return None

    # Przypadek 2: lista
    if isinstance(raw, list):
        if not raw:
            print(f"  [{label}] Uwaga: w pliku {path.name} lista jest pusta.")
            return None
        first = raw[0]
        if isinstance(first, dict) and "pozycje_oferty" in first:
            return first
        else:
            print(
                f"  [{label}] Uwaga: w pliku {path.name} lista[0] nie jest dictem z 'pozycje_oferty'. "
                f"typ(list[0])={type(first)}"
            )
            return None

    print(f"  [{label}] Uwaga: w pliku {path.name} JSON ma nieobsługiwany typ: {type(raw)}")
    return None


def _compare_items(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n_gold = len(gold_items)
    n_pred = len(pred_items)
    n = min(n_gold, n_pred)

    field_matches = {f: 0 for f in FIELDS_TO_COMPARE}
    field_total = {f: 0 for f in FIELDS_TO_COMPARE}

    examples: Dict[str, List[Tuple[int, Any, Any]]] = {f: [] for f in FIELDS_TO_COMPARE}

    for idx in range(n):
        g = gold_items[idx]
        p = pred_items[idx]
        for field in FIELDS_TO_COMPARE:
            g_val = g.get(field, "")
            p_val = p.get(field, "")
            field_total[field] += 1
            if g_val == p_val:
                field_matches[field] += 1
            else:
                if len(examples[field]) < 3:  # max 3 przykłady różnic na pole
                    examples[field].append((idx, g_val, p_val))

    return {
        "n_gold": n_gold,
        "n_pred": n_pred,
        "field_matches": field_matches,
        "field_total": field_total,
        "examples": examples,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    gold_dir = base_dir / GOLD_JSON_DIR
    pred_dir = base_dir / LLM_OUTPUT_DIR

    if not gold_dir.exists():
        print(f"❌ Katalog gold_json nie istnieje: {gold_dir}")
        return
    if not pred_dir.exists():
        print(f"❌ Katalog LLM_output nie istnieje: {pred_dir}")
        return

    gold_files = sorted(gold_dir.glob("*.json"))
    if not gold_files:
        print(f"❌ Brak plików *.json w {gold_dir}")
        return

    print(f"Znaleziono {len(gold_files)} gold JSONów.\n")

    for gold_path in gold_files:
        stem = gold_path.stem  # np. "1_144_weidmuller"
        pred_path = pred_dir / f"{stem}.json"

        print(f"=== {stem} ===")
        if not pred_path.exists():
            print(f"  Brak pliku predykcji: {pred_path}")
            print()
            continue

        try:
            raw_gold = _load_json(gold_path)
            raw_pred = _load_json(pred_path)
        except Exception as e:
            print(f"  ❌ Błąd przy wczytywaniu JSONów: {e}")
            print()
            continue

        gold_obj = _normalize_offer_object(raw_gold, gold_path, "gold")
        pred_obj = _normalize_offer_object(raw_pred, pred_path, "pred")

        if gold_obj is None or pred_obj is None:
            print("  ❌ Nie udało się znormalizować struktur (brak 'pozycje_oferty').")
            print()
            continue

        gold_items = gold_obj.get("pozycje_oferty", [])
        pred_items = pred_obj.get("pozycje_oferty", [])

        if not isinstance(gold_items, list) or not isinstance(pred_items, list):
            print("  ❌ 'pozycje_oferty' nie jest listą w którymś z plików")
            print()
            continue

        result = _compare_items(gold_items, pred_items)

        print(f"  Liczba pozycji: gold={result['n_gold']}, pred={result['n_pred']}")
        for field in FIELDS_TO_COMPARE:
            total = result["field_total"][field]
            if total == 0:
                acc = 0.0
            else:
                acc = result["field_matches"][field] / total * 100.0
            print(f"    {field:16s}: {acc:5.1f}% ({result['field_matches'][field]}/{total})")

        print("  Przykłady różnic:")
        for field in FIELDS_TO_COMPARE:
            ex = result["examples"][field]
            if not ex:
                continue
            print(f"    - {field}:")
            for idx, g_val, p_val in ex:
                print(f"      idx {idx}:")
                print(f"        gold: {g_val!r}")
                print(f"        pred: {p_val!r}")

        print()


if __name__ == "__main__":
    main()
