# PDF Offer Processor

A Python application for processing PDF offers using Google Gemini AI to extract structured data.

## Project Structure

```
.
├── main.py           # Entry point - handles PDF input/output
├── processor.py      # Core processing logic (3-phase pipeline)
├── models.py         # Pydantic data models
├── prompts.py        # AI prompt templates
└── README.md         # This file
```

### File Descriptions

**`main.py`** - Entry point
- Loads PDF from hardcoded path
- Manages the processing workflow
- Saves results to JSON
- Displays summary

**`processor.py`** - Core processing logic
- `process_pdf()`: Main 3-phase pipeline
- Phase 1: Extract header + lightweight item list
- Phase 2: Batch process items in detail
- Phase 3: Assemble and validate results
- Helper functions for retry logic, business logic, etc.

**`models.py`** - Data models
- `Unit`: Material units (SZT, m, l, pal)
- `WeightUnit`: Weight units (KG, G)
- `Language`: Supported languages (PL, EN, DE)
- `OfferItemText`: Translation structure
- `OfferItem`: Individual item model
- `ItemPreview`: Lightweight item preview
- `OfferScan`: Phase 1 scan result

**`prompts.py`** - Prompt templates
- `get_base_prompt()`: Common base prompt
- `get_phase1_prompt()`: Header and item list extraction
- `get_phase2_prompt()`: Detailed item extraction

## Setup

### Prerequisites
- Python 3.10+
- Google API key with Gemini access

### Installation

1. Clone or download the project files

2. Install dependencies:
```bash
pip install google-genai pydantic
```

3. Set environment variables:
```bash
export GOOGLE_API="your-google-api-key"
export GEMINI_MODEL="gemini-2.5-pro"  # or gemini-2.5-flash, etc.
```

Or create a `.env` file (requires python-dotenv):
```
GOOGLE_API=your-google-api-key
GEMINI_MODEL=gemini-2.5-pro
```

### Configuration

Edit `main.py` to set the PDF paths:

```python
# Path to the PDF file to process (hardcoded)
PDF_INPUT_PATH = "/path/to/your/offer.pdf"

# Path where the output JSON will be saved
OUTPUT_PATH = "/path/to/output.json"
```

## Usage

Run the processor:

```bash
python main.py
```

The script will:
1. Load the PDF from the hardcoded path
2. Extract offer header and items (Phase 1)
3. Process items in batches (Phase 2)
4. Assemble and validate results (Phase 3)
5. Save results to JSON
6. Display a summary

## Output Format

The output JSON contains an array with a single offer object:

```json
[
  {
    "id": "1234567890",
    "nazwa_dostawcy": "Supplier Name",
    "data_oferty": "2024-01-15",
    "waluta_ceny_oferty": "PLN",
    "liczba_elementow": 5,
    "pozycje_oferty": [
      {
        "id": "1",
        "id_oferty": "1234567890",
        "opis": "Original description",
        "grupa_materialow": "Group",
        "numer_oem": "OEM-123",
        "producent": "Manufacturer",
        "serial_numbers": "CN:123;PKWiU:456",
        "tłumaczenia": [
          {
            "item_language": "PL",
            "item_description": "Polish description"
          },
          {
            "item_language": "EN",
            "item_description": "English description"
          },
          {
            "item_language": "DE",
            "item_description": "German description"
          }
        ],
        "kwota_ceny_oferty": 99.99,
        "kwota_ceny_zniżka_kwota": 10.00,
        "kwota_ceny_zniżka_procent": 0.0,
        "gross_weight": 1.5,
        "net_weight": 1.2,
        "unit_of_weight": "KG"
      }
    ]
  }
]
```

## Processing Phases

### Phase 1: Scan
- Extracts offer header (ID, supplier, date, etc.)
- Creates lightweight list of all items
- Excludes services, transport, and non-physical goods
- Returns: `OfferScan` object with `items_preview`

### Phase 2: Batch Processing
- Processes items in batches of 6
- Extracts complete details for each item
- Includes retry logic (full batch → half batches → individual items)
- Returns: List of `OfferItem` objects

### Phase 3: Assembly & Validation
- Combines scan data with processed items
- Applies business logic:
  - Fills missing manufacturer from supplier name
  - Calculates discount amounts
  - Cleans OEM duplicates from serial numbers
  - Validates weight units
  - Sets default availability
- Returns: Final offer structure

## Error Handling

The processor includes:
- Batch-level retry logic (splits into half batches)
- Item-level retry logic (3 attempts)
- Graceful degradation (continues with available items)
- Progress tracking via `StatusTracker`

## Customization

### Modify Prompts
Edit `prompts.py` to customize extraction instructions for Phase 1 and Phase 2.

### Change Batch Size
In `processor.py`, modify:
```python
BATCH_SIZE = 6  # Change this value
```

### Add Custom Business Logic
Extend `_apply_business_logic_to_item()` in `processor.py`.

### Adjust Retry Strategy
Modify the retry logic in `_process_single_batch()` function.

## Troubleshooting

**"API key is empty"**
- Set the `GOOGLE_API` environment variable

**"No items found in Phase 1"**
- The PDF may not contain any valid offer items
- Check if items are being filtered out as services/transport

**"API usage exceeded"**
- Reduce batch size or implement rate limiting
- Wait before running again

**"Invalid weight unit"**
- Check the source PDF for weight unit formats
- Modify `VALID_WEIGHT_UNITS` in `processor.py` if needed

## API Costs

This application makes API calls to Google's Gemini model:
- Phase 1: 1 call for header + item list
- Phase 2: Multiple calls based on item count and batch size
- With Thinking enabled: Higher token usage on gemini-2.5-pro/flash

Monitor your usage in the Google AI Studio dashboard.

## License

[Add your license here]

## Support

[Add support information here]
