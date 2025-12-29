"""
Prompt templates for PDF offer processing (LLaMA, JSON-only).

All task instructions are in English; extracted content (e.g. 'opis')
must stay in the original language of the PDF (typically Polish).
"""


def get_base_prompt() -> str:
    """
    Shared part of the prompt for all phases.
    Instructions cover only the semantic content; JSON formatting is enforced by Outlines.
    """
    return """
You extract structured information from PDF commercial offers.

CONTEXT:
- The offers are directed to a specific buyer (for example a production company).
- Supplier fields always describe the company that SENDS the offer.
- The input you receive is plain text extracted from a PDF file.

GENERAL RULES:
- Work ONLY on text between [PDF_TEXT_START] and [PDF_TEXT_END].
- Ignore all service-type items:
  transport, installation, inspections, maintenance, travel costs,
  surcharges, "Dopłata", "Praca serwisu", etc.
- Extract ONLY physical goods (products, components, materials).
- Keep all product descriptions ("opis") in the original language from the PDF (usually Polish).
  Do NOT translate to English.

MISSING DATA RULES:
- If a field is missing:
  - textual  -> "",
  - date     -> null,
  - numeric  -> 0 or 0.0.

VERY IMPORTANT – DO NOT GUESS:
- If you are not sure that a value is present in the text, leave the corresponding field
  empty ("") or 0.0 / null.
- An empty field is better than an invented or incorrect value.
"""


def get_phase1_prompt(base_prompt: str) -> str:
    """
    Phase 1 – extract only items_preview from product tables.
    Header fields are ignored and may stay empty.
    """
    return f"""{base_prompt}

PHASE 1 TASK: ITEMS PREVIEW FROM PRODUCT TABLES (NO HEADER FIELDS)

Your goal in this phase is to build only the 'items_preview' list.

Ignore the offer header completely:
- do NOT analyse offer numbers, dates, supplier names, addresses,
  logos, sign-off blocks or contact information.
- you may leave "id", "nazwa_dostawcy" and "data_oferty" empty or null.

Focus only on product tables and enumerated lists that describe PHYSICAL GOODS.
Treat each row of a product table as one potential item if it describes a material,
component or product.

For the OfferScan schema:
- "id": set to "" if unknown.
- "nazwa_dostawcy": set to "" if unknown.
- "data_oferty": set to null if unknown.
- "items_preview": must contain physical goods found in the text.

TABLE AND HEADER RECOGNITION (IMPORTANT):

First try to identify table headers with column names. Typical examples:

- "Lp", "L.p.", "Poz." – position number (ignore in description).
- "Kod produktu", "Kod", "Nr artykułu", "Numer artykułu", "Symbol", "ID" – product code column.
- "Nazwa", "Nazwa produktu", "Nazwa towaru/usługi" – product name column.
- "Ilość", "Ilość j.m.", "Ilość [szt.]", "j.m.", "jm." – quantity and unit (not required in description).
- "Cena/sztukę", "Cena jednostkowa", "Cena przed rabatem", "Cena netto" – unit price.
- "Cena końcowa", "Cena", "Wartość netto", "Wartość brutto" – total value.

Columns such as:
- "Dostępność", "Rabat %", "VAT", "Stawka VAT", "Wartość brutto", "Uwagi"
are additional information that you do NOT include in the product description.

For each product row:
- Build the description mainly from the product code and product name columns if they exist.
- Quantity and prices are NOT required in the description.
- You may ignore pure service lines, notes and remarks.

DESCRIPTION FIELD ("opis"):

For each physical product row in a table create one object in "items_preview" with:
- "position_id": sequential integer 1, 2, 3, ...
- "opis": short description based on the whole table row:
  • If there is a dedicated product code column ("Kod produktu", "Nr artykułu", "Symbol", "ID", etc.),
    ALWAYS keep this code together with the product name in the description.
  • If there is no dedicated product code column, but a clear code appears at the very beginning of the row
    before the product name (e.g. "ABC-123 - Moduł wejściowy 24V DC"), you MAY keep that code.
  • If you do NOT see any clear product code in the row, describe only the product name/type.
    NEVER invent a product code.
  • You may omit quantity and prices from the description, but do NOT remove an existing product code if it is present.
  • examples:
    "140017 EFB O-ring"
    "100,37036382 Driven roller"
    "8,21009886 Roller chain 1/2\""

NOTE:
In Phase 1 you may keep product codes at the start of "opis".
These codes will later be extracted into "numer_oem" during Phase 2,
so do not attempt to remove or separate them here.

UNIT PRICE FIELD ("kwota_ceny_oferty"):

- "kwota_ceny_oferty": unit price as a number (float), or 0.0 if you cannot find it.
- Use unit price from columns like "Cena/sztukę", "Cena jednostkowa", "Cena netto",
  not the total value ("Cena końcowa", "Wartość netto").
- If you are not sure which number is the unit price, set "kwota_ceny_oferty" to 0.0.

OUTPUT SIZE NOTE:
- If the file contains an extremely large number of rows (hundreds),
  you may skip clearly empty or duplicated rows, but do NOT skip valid product lines.

SERVICE LINES TO IGNORE COMPLETELY (DO NOT CREATE ITEMS):
- transport, shipment, delivery, forwarding (e.g. "Koszt spedycji", "Koszt transportu")
- courier costs ("dostawa kurierska", "GLS", "UPS")
- installation, commissioning, configuration
- inspections, maintenance, service work, travel costs
- generic handling fees ("opłata manipulacyjna", "handling costs")
- freight / forwarding costs even if they include a code-like token (e.g. "Koszt spedycji 030G")


If a row description starts with or contains mainly these phrases,
treat it as a service and SKIP it.

Work ONLY on the text between [PDF_TEXT_START] and [PDF_TEXT_END].

The final JSON must have exactly this top-level structure:

{{
  "id": "",
  "nazwa_dostawcy": "",
  "data_oferty": null,
  "items_preview": [
    {{
      "position_id": 1,
      "opis": "...",
      "kwota_ceny_oferty": 0.0
    }}
  ]
}}

You may fill "id", "nazwa_dostawcy" and "data_oferty" as described above,
but the main goal is a correct and complete "items_preview" list.
Return only ONE JSON object and nothing else.
"""


def get_phase2_prompt(base_prompt: str, batch_items: list, batch_start: int) -> str:
    """
    Phase 2 – detailed item extraction.
    JSON formatting is enforced by Outlines; prompt contains only substantive instructions.
    """
    item_count = len(batch_items)
    first_idx = batch_start + 1
    last_idx = batch_start + item_count

    batch_prompt = f"""{base_prompt}

PHASE 2 TASK: DETAILED ITEMS (MINIMAL OfferItem SCHEMA)

You will receive:
- the same offer text,
- a subset of items from Phase 1 ("items_preview").

For EACH of these preview items produce one fully detailed object
according to the OfferItem schema with EXACTLY the following fields:

- "id": primary key for the item, use a simple running counter "1", "2", "3", ...
- "id_oferty": offer ID for all items in this file (if unknown, use an empty string "").
- "opis": original description of the item, based on the PDF text (do not translate).
- "grupa_materialow": material group or product category inferred from the context.
- "numer_oem": main OEM / article / item number for this product.
- "producent": manufacturer name.
- "serial_numbers": other identification or classification codes (CN, PKWiU, indices, etc.).

GENERAL RULES FOR ALL FIELDS:
- Use only information that appears in the offer text between [PDF_TEXT_START] and [PDF_TEXT_END].
- Do NOT invent values that are not supported by the text.
- If a field is missing, use an empty string "".
- Prefer empty fields over guessed or incorrect values.

VERY IMPORTANT – MISSING DATA:
- If the offer text does not clearly specify an offer ID, OEM number, manufacturer,
  or other codes, leave the corresponding field empty ("").
- Empty is BETTER than incorrect. Never fabricate a product code or manufacturer.

DESCRIPTION FIELD ("opis"):
- Do NOT append price, discounts, totals or other numeric columns to "opis".
- Do NOT copy "kwota_ceny_oferty" into "opis".
- "opis" must describe ONLY the product itself (name, type, variant, parameters) and MUST NOT contain the OEM/article number.
- If the preview opis begins with a product code-like token
  (short alphanumeric code, often with dashes/slashes),
  ALWAYS remove it from "opis" and put it into "numer_oem".
- If the leading token is clearly NOT a product code (e.g. "LP", "JM", "szt.", "Cena"),
  then do NOT treat it as numer_oem.
- After removing the OEM from the start, keep everything else exactly as in the PDF (no translation, no rewriting).

TABLE STRUCTURE AND COLUMN MAPPING:

Whenever possible, use the table headers to map columns to fields. Typical mappings:

- "Kod produktu", "Kod", "Nr artykułu", "Numer artykułu", "Symbol", "ID"
  → candidate for "numer_oem".
- "Nazwa", "Nazwa produktu", "Nazwa towaru/usługi"
  → main product name used in "opis".
- "CN", "PKWiU", "Indeks", "Nr magazynowy"
  → candidates for "serial_numbers" (classification or internal codes).

Examples of table headers:
- "Lp Kod produktu Nazwa produktu Ilość Cena/sztukę Cena końcowa"
  • numer_oem from "Kod produktu"
  • opis built mainly from "Nazwa produktu"
- "Poz. Nr artykułu Ilość j.m. Cena jednostkowa Cena"
  • numer_oem from "Nr artykułu"
  • opis built from product name / description column
- "Lp. Nazwa towaru/usługi Dostępność Ilość jm. Cena przed rabatem Rabat % Cena netto Wartość netto VAT"
  • typically NO dedicated product code column; in such case numer_oem may be empty if no clear code is present.

Never treat columns such as "Dostępność", "Rabat %", "VAT", "Cena", "Wartość netto", "Wartość brutto"
as sources of product codes.

KEY FIELDS (MUST NOT BE GUESSED):

ADDITIONAL RULES FOR PRODUCT CODES (IMPORTANT):

- If the table contains labels like:
  "Nr produktu", "Nr produktu SICK", "Indeks", "Code", "Item code", "Material", "Article No."
  treat the value as the main product code -> put it into "numer_oem".
- Do NOT put such product codes into "serial_numbers".

1) numer_oem

- This is the MAIN product code (OEM / article / item number).
- Prefer values from dedicated columns with headers like:
  "Kod produktu", "Kod", "Nr artykułu", "Numer artykułu", "Symbol", "ID".
- If there is NO dedicated column, you may treat as numer_oem a short alphanumeric code
  at the very beginning of the row, placed before the product name and separated by a space or dash,
  for example: "ABC-123 - Moduł wejściowy 24V DC".
- NEVER assign CN, PKWiU, HS, tariff numbers, or any long numeric codes (8+ digits) to "numer_oem".
- If a row contains:
  • a short article number (e.g. "1119902", "3RH2911-1HA22") AND
  • classification codes ("CN: 85369010", "PKWiU 27.33.13.0"),
  then:
  • numer_oem = short article number
  • serial_numbers = "CN: ...;PKWiU: ..."
- If you are NOT sure that a value is a product code, set "numer_oem" to an empty string "".
- If the preview opis contains a product code at the beginning
  (e.g. "ABC-123 - Opis produktu", "4500123 O-ring"),
  this code must be extracted into "numer_oem" and removed from "opis".
- The final opis must NEVER contain the OEM/article number.


STRICT EXCLUSIONS FOR serial_numbers (MANDATORY):
- serial_numbers MUST NOT contain:
  prices (PLN/EUR), discounts (Rabat), VAT, totals (Wartość),
  quantities (Ilość, J.m., szt.), delivery/availability (na stanie, termin realizacji),
  URLs (http/https/www), or any logistics notes.
- If you see only such information and no real codes, set serial_numbers to "".

2) serial_numbers

- This field collects all identification or classification codes related to the item
  that are NOT the main OEM/article number.
- Typical examples:
  • customs codes (e.g. "CN 85369010"),
  • PKWiU codes,
  • internal warehouse numbers or catalogue indices,
  • any additional structured codes that clearly refer to this item.
- If a header contains words "CN", "PKWiU", "Indeks", "Nr magazynowy", etc.,
  you may treat the values from that column as elements of "serial_numbers".
- Merge all such codes into ONE string separated by semicolons, for example:
  "CN: 85369010;PKWiU: 27.33.13.0;ID: 123456".
- Always add prefixes like "CN:" or "PKWiU:" when the code type is clear.
- Do NOT include pure technical parameters such as voltage, current, power, frequency,
  IP rating, dimensions, weight, etc.
- Do NOT duplicate numer_oem here (even if the same number appears again with a different label).
- If you do not see any such codes, set "serial_numbers" to "".




OTHER FIELDS:

- id:
  • Use a simple integer-like string "1", "2", "3", ...
  • The order should follow the order of items in Phase 1.

- id_oferty:
  • If an offer ID is visible in the header or in a dedicated field, copy it exactly.
  • If multiple IDs are present, use the one that clearly identifies the offer document.
  • If you cannot find any offer ID, use an empty string "".

  To extract id_oferty, search the header for typical patterns:
  - "Numer / Data wystawienia: <ID> / <date>"
  - "Oferta Nr <ID>"
  - "Offer No.: <ID>"
  - "Numer Oferty / data <ID>"
  If multiple values look like offer IDs, choose the one explicitly labeled as the offer number.
  If none exists, set "".

- grupa_materialow:
  • Assign a short, general category based on the description and context of the item.
  • Examples of categories: "złączki", "kable", "wyłączniki", "bezpieczniki",
    "akcesoria montażowe", "automatyka", "mechanika", "inne".
  • If you are not sure, choose the closest reasonable category.

- producent:
  • Use the manufacturer name present directly in the item line or clearly associated with the product
    (e.g. SIEMENS, LAPP, SICK, KRONES).
  • DO NOT copy the supplier name (distributor) into "producent" unless the PDF explicitly states
    that the supplier is also the manufacturer.
  • If no manufacturer can be determined with certainty, set producent to "".

STRICT CARDINALITY CONSTRAINT (CRUCIAL):

You are given exactly {item_count} preview items (Item {first_idx} to Item {last_idx}).
For EACH input item you MUST return EXACTLY ONE OfferItem object.

- The output JSON ARRAY MUST contain exactly {item_count} elements.
- The order of objects in the output array MUST match the order of input items
  (first object corresponds to Item {first_idx}, last object to Item {last_idx}).
- Do NOT merge different items into one object.
- Do NOT create any additional or synthetic items.
- Do NOT repeat the same item multiple times.
- If you are uncertain, still return exactly one best-effort object per input item,
  but never invent fields that are not visible in the text.

ITEMS TO PROCESS:
"""

    for idx, item in enumerate(batch_items, start=batch_start + 1):
        batch_prompt += (
            f"\nItem {idx} (preview):\n"
            f"  opis: {item['opis']}\n"
            f"  kwota_ceny_oferty: {item['kwota_ceny_oferty']}\n"
        )

    batch_prompt += f"""

OUTPUT FORMAT (MANDATORY):

Provide the result as a JSON ARRAY of OfferItem objects following the schema above.
The array MUST have exactly {item_count} elements and MUST be the ONLY thing in your answer.
Return only ONE JSON array and NO additional text, comments or explanations.
"""

    return batch_prompt
