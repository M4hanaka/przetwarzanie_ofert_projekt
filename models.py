import enum
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, conlist, field_validator
import re

class Unit(enum.Enum):
    """Supported units for quantity."""

    sztuki = "SZT"
    metry = "m"
    litry = "l"
    palety = "pal"


class WeightUnit(enum.Enum):
    """Supported units for weight."""

    kilogramy = "KG"
    gramy = "G"


class Language(enum.Enum):
    """Language codes for item translations."""

    polski = "PL"
    angielski = "EN"
    niemiecki = "DE"

class OfferItemText(BaseModel):
    """
    Short translated description for an item.

    Used for building compact item names (e.g. for ERP) that must fit into
    a strict character limit and avoid OEM/serial/manufacturer codes.
    """

    item_language: Language
    item_description: str = Field(
        ...,
        max_length=39,
        description=(
            "Short item name that fits in 40 characters. "
            "You can avoid using commas. "
            "Do NOT include manufacturer name, model name, OEM number, serial number, "
            "or customs/PKWiU codes. "
            "Keep only a general name and the most important technical parameters."
        ),
    )

class MaterialGroup(enum.Enum):
    zlaczki = "złączki"
    przylaczki = "przyłączki"
    mufki = "mufki"
    kable = "kable"
    wylaczniki = "wyłączniki"
    automatyka = "automatyka"
    mechanika = "mechanika"
    filtry = "filtry"
    uszczelki = "uszczelki"
    uszczelnienia = "uszczelnienia"
    armatura = "armatura"
    akcesoria_montazowe = "akcesoria montażowe"
    oprogramowanie = "oprogramowanie"
    inne = "inne"

class OfferItem(BaseModel):
    """
    OfferItem – individual item within an offer.

    In the project exactly 7 key fields are used:
    id, id_oferty, opis, grupa_materialow, numer_oem, producent, serial_numbers.

    Fields numer_oem and serial_numbers are important, but:
    - they MUST NOT be guessed,
    - it is allowed and expected to leave them empty ("") when the offer text
      does not provide a clear value.
    """

    id: str = Field(
        ...,
        description=(
            "Primary key for the offer item. "
            "Use a simple running counter like '1', '2', '3', ... "
            "in the order in which items appear in the offer."
        ),
    )

    id_oferty: str = Field(
        ...,
        description=(
            "Foreign key to the parent offer (offer ID). "
            "If the offer ID appears in the header or on each page, copy it exactly. "
            "If no offer ID is available in the text, use an empty string ''. "
            "Never invent an artificial ID."
        ),
    )

    opis: str = Field(
        ...,
        description=(
            "Original item description taken from the offer, usually in Polish. "
            "Do NOT translate it to English. "
            "Keep the original sentence or line that describes this position. "
            "You may merge multi-line descriptions into one sentence, but do not remove "
            "important technical details. Do not append prices, discounts, totals, or VAT."
        ),
    )

    grupa_materialow: str = Field(
        ...,
        description=(
            "Material group/category. Use one of the predefined values. "
            "If unsure, choose the closest general category."
        ),
    )


    numer_oem: str = Field(
        ...,
        description=(
            "Main item/OEM/article number for this product. "
            "It is usually an alphanumeric code assigned by the manufacturer or supplier. "
            "Typical examples: 'TRP 24VDC 1CO', 'WDU 2.5', '4510022', '5SL4102-6'. "
            "It often appears near the beginning of the line or in a dedicated column "
            "such as 'Numer artykułu', 'Kod produktu', 'Symbol', 'ID'. "
            "Do NOT use classification codes here (CN, PKWiU, customs codes) "
            "and do NOT use invoice numbers, line numbers, prices, or quantities. "
            "If multiple numbers are visible, choose the one that clearly identifies "
            "this item as a product code. "
            "If you are not sure that a value is the correct OEM/article number, "
            "leave this field empty (''). Never guess."
        ),
    )

    producent: str = Field(
        ...,
        description=(
            "Manufacturer name for the item, for example SIEMENS, SCHNEIDER ELECTRIC, "
            "WEIDMULLER, LAPP, PHOENIX CONTACT. "
            "Use the name visible in the offer header, logo, or item description. "
            "Do NOT guess the manufacturer if it is not clearly stated in the text. "
            "If the manufacturer cannot be identified, use an empty string ''."
        ),
    )

    serial_numbers: str = Field(
        ...,
        description=(
            "All identification, classification, or catalog numbers related to this item "
            "that are NOT the main OEM/article number. "
            "Examples: customs codes (CN), PKWiU codes, internal ID codes used by the supplier, "
            "warehouse numbers, or other catalogue identifiers. "
            "Merge all such codes into a single string separated by semicolons, for example: "
            "'CN: 85369010;PKWiU: 27.33.13.0;ID: 123456'. "
            "Always add prefixes like 'CN:' or 'PKWiU:' where applicable. "
            "Do NOT include technical parameters such as voltage, current, power, frequency, "
            "dimensions, IP ratings, etc. "
            "Do NOT duplicate numer_oem here. "
            "If there are no such additional codes in the text, leave this field empty (''). "
            "Never invent codes that are not explicitly present."
        ),
    )

    _ALLOWED_GROUPS = {
        "złączki", "przyłączki", "mufki", "kable", "wyłączniki",
        "automatyka", "mechanika", "filtry", "uszczelki", "uszczelnienia",
        "armatura", "akcesoria montażowe", "oprogramowanie", "inne",
    }

    _GROUP_ALIASES = {
        "przewody": "kable",
        "uszczelka": "uszczelki",
        "uszczelnienie": "uszczelnienia",
        "monitoring": "automatyka",
    }

    @field_validator("grupa_materialow", mode="before")
    @classmethod
    def validate_grupa_materialow(cls, v):
        if not isinstance(v, str):
            return "inne"
        s = v.strip()
        s = cls._GROUP_ALIASES.get(s, s)
        return s if s in cls._ALLOWED_GROUPS else "inne"

    @field_validator("serial_numbers", mode="before")
    @classmethod
    def validate_serial_numbers(cls, v):
        return _clean_serial_numbers(v if isinstance(v, str) else "")

    @field_validator("opis", mode="before")
    @classmethod
    def validate_opis(cls, v):
        return _clean_opis(v if isinstance(v, str) else "")

_PRICE_OR_NOISE_RE = re.compile(
    r"(?i)\b("
    r"pln|eur|vat|rabat|cena|wartość|ilość|termin|dostaw|na stanie|"
    r"szt\.|j\.m\.|jm|kg|g|dni|%|https?://|www\."
    r")\b"
)

def _clean_serial_numbers(value: str) -> str:
    if not value:
        return ""
    v = value.strip()

    # Jeśli zawiera ewidentny szum (ceny, rabaty, URL, terminy) – wyczyść do pustego
    # (w tych ofertach i tak zwykle nie było CN/PKWiU, tylko śmieci).
    if _PRICE_OR_NOISE_RE.search(v):
        # wyjątek: pozwól zachować CN/PKWiU/EAN/ID jeśli są podane poprawnie
        parts = [p.strip() for p in v.split(";") if p.strip()]
        keep = []
        for p in parts:
            if re.match(r"(?i)^(CN:\s*\S+|PKWiU:\s*\S+|EAN:\s*\S+|ID:\s*\S+)$", p):
                keep.append(p)
        return ";".join(keep)
    return v


def _clean_opis(value: str) -> str:
    if not value:
        return ""
    v = value.strip()
    # Nie dopuszczaj ceny/rabatu/URL w opisie
    if _PRICE_OR_NOISE_RE.search(v):
        # minimalna sanitizacja: usuń fragmenty po typowych separatorach
        # (nie zgadujemy, tylko tniemy śmieci)
        v = re.split(r"(?i)\b(pln|eur|vat|rabat|cena|wartość|https?://|www\.)\b", v)[0].strip()
    return v

    
class ItemPreview(BaseModel):
    """
    Lightweight item preview for Phase 1.

    Used as an intermediate result from the initial table scan:
    - 'opis' is a short row-level description (code + name + key info),
    - 'kwota_ceny_oferty' is the unit price if clearly available, otherwise 0.0.
    """

    position_id: int = Field(
        ...,
        description=(
            "Sequential integer position within the offer table (1, 2, 3, ...), "
            "following the order of rows in the PDF."
        ),
    )
    opis: str = Field(
        ...,
        description=(
            "Short description based on the entire table row. "
            "Typically contains the product code (if present) and the product name. "
            "It should not contain totals, VAT, or explanatory notes."
        ),
    )
    kwota_ceny_oferty: float = Field(
        ...,
        description=(
            "Unit price for this item as a numeric value. "
            "Use the value from columns like 'Cena/sztukę', 'Cena jednostkowa', 'Cena netto'. "
            "If the unit price cannot be reliably identified, set this field to 0.0."
        ),
    )


class OfferScan(BaseModel):
    """
    Phase 1 scan result with offer header and item previews.

    This is the top-level object produced after the initial scan over the PDF text.
    It contains basic header fields (which may be empty) and a list of all item previews.
    """

    id: str = Field(
        ...,
        description=(
            "Offer ID as it appears in the document header (e.g. '198/12/2024', "
            "'OF/PiT/MMG/2025/03748'). "
            "If no clear offer ID is visible, use an empty string ''."
        ),
    )
    nazwa_dostawcy: str = Field(
        ...,
        description=(
            "Supplier name (the company that sends the offer). "
            "If the supplier name cannot be determined from the text, use an empty string ''."
        ),
    )
    data_oferty: Optional[str] = Field(
        default=None,
        description=(
            "Offer date in textual form (e.g. '2024-12-27', '30.04.2024'), "
            "or null if the date is not present in the document."
        ),
    )
    items_preview: List[ItemPreview] = Field(
        default_factory=list,
        description=(
            "Lightweight list of all item previews (one per physical product row) "
            "detected in the Phase 1 scan."
        ),
    )
