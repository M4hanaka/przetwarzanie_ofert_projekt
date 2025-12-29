# main.py
import sys
from processor import run_single_pdf, run_batch_dir
from pdf_to_text import PdfToTextConfig

MODE = "2"  # "1" single, "2" batch

# SINGLE
PDF_INPUT_PATH = "/home/karol_zych/OFERTY/oferty_pdf_1/1_144_weidmuller.pdf"
OUTPUT_PATH = "/home/karol_zych/przetwarzanie_ofert_projekt/OUTPUT/out.json"

# BATCH
BATCH_INPUT_DIR = "/home/karol_zych/OFERTY/oferty_pdf_1" # _1 /or/ _2
VIEW_TEXT_DIR = "OUTPUT/view_text"
BATCH_OUTPUT_DIR = "OUTPUT/LLM_output"


def main() -> int:
    text_cfg = PdfToTextConfig(max_pages=8, max_chars=40_000, include_page_headers=True)

    if MODE.strip() in ("2", "batch"):
        return run_batch_dir(
            input_dir=BATCH_INPUT_DIR,
            output_dir=BATCH_OUTPUT_DIR,
            view_text_dir=VIEW_TEXT_DIR,
            text_cfg=text_cfg,
            ocr=None,
            only_missing=False,
        )

    return run_single_pdf(
        pdf_path=PDF_INPUT_PATH,
        output_json_path=OUTPUT_PATH,
        text_cfg=text_cfg,
        ocr=None,
        save_preview_txt=False,
    )


if __name__ == "__main__":
    sys.exit(main())
