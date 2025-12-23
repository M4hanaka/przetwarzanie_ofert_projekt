import json
from pathlib import Path

def fix_file(in_path: Path, out_path: Path) -> int:
    issues_fixed = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON decode error in {in_path}:{line_no}: {e}")

            phase = obj.get("phase")
            bi = obj.get("batch_index", None)

            # Normalizacja:
            # - phase1: batch_index zawsze 0 (walidator wymaga int>=0)
            # - phase2: batch_index musi być int>=0 (jeżeli brak/None -> ustaw 0, ale to powinno występować rzadko)
            if phase == "phase1":
                if not isinstance(bi, int) or bi < 0:
                    obj["batch_index"] = 0
                    issues_fixed += 1
            elif phase == "phase2":
                if not isinstance(bi, int) or bi < 0:
                    obj["batch_index"] = 0
                    issues_fixed += 1
            else:
                # Jeżeli kiedyś dodasz inne fazy, też wymuś poprawność:
                if not isinstance(bi, int) or bi < 0:
                    obj["batch_index"] = 0
                    issues_fixed += 1

            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return issues_fixed

def main():
    base = Path("/home/karol_zych/przetwarzanie_ofert_projekt/INPUT/train_json")
    train_in = base / "train.jsonl"
    val_in = base / "val.jsonl"

    train_out = base / "trainn.jsonl"
    val_out = base / "vall.jsonl"

    tf = fix_file(train_in, train_out)
    vf = fix_file(val_in, val_out)

    print(f"Saved: {train_out} (fixed {tf} records)")
    print(f"Saved: {val_out}   (fixed {vf} records)")

if __name__ == "__main__":
    main()
